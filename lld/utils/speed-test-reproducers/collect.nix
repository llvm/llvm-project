#===-----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//
#
# This is a Nix recipe for collecting reproducers for benchmarking purposes in a
# reproducible way. It works by injecting a linker wrapper that embeds a
# reproducer tarball into a non-allocated section of every linked object, which
# generally causes them to be smuggled out of the build tree in a section of the
# final binaries. In principle, this technique should let us collect reproducers
# from any project packaged by Nix without project-specific knowledge, but as
# you can see below, many interesting ones need a few hacks.
#
# If you have Nix installed, you can build the reproducers with the following
# command:
#
# TMPDIR=/var/tmp nix-build -j6 --log-format bar collect.nix
#
# This will result in building several large projects including Chromium and
# Firefox, which will take some time, and it will also build most of the
# dependencies for non-native targets. Eventually you will get a result
# directory containing all the reproducers.
#
# The following projects have been tested successfully:
# - chrome (native only, cross builds fail building the qtbase dependency)
# - firefox (all targets)
# - linux-kernel (all targets, requires patched nixpkgs)
# - ladybird (native only, same problem as chromium)
# - llvm (all targets)

{
  nixpkgsDir ? fetchTarball "https://github.com/NixOS/nixpkgs/archive/992f916556fcfaa94451ebc7fc6e396134bbf5b1.tar.gz",
  nixpkgs ? import nixpkgsDir,
}:
let
  reproducerPkgs =
    crossSystem:
    let
      pkgs = nixpkgs { inherit crossSystem; };
      # Wraps the given stdenv and lld package into a variant that collects
      # the reproducer and builds with debug info.
      reproducerCollectingStdenv =
        stdenv: lld:
        let
          bintools = stdenv.cc.bintools.override {
            extraBuildCommands = ''
              wrap ${stdenv.cc.targetPrefix}nix-wrap-lld ${nixpkgsDir}/pkgs/build-support/bintools-wrapper/ld-wrapper.sh ${lld}/bin/ld.lld
              export lz4=${pkgs.lib.getBin pkgs.buildPackages.lz4}/bin/lz4
              substituteAll ${./ld-wrapper.sh} $out/bin/${stdenv.cc.targetPrefix}ld
              chmod +x $out/bin/${stdenv.cc.targetPrefix}ld
              substituteAll ${./ld-wrapper.sh} $out/bin/${stdenv.cc.targetPrefix}ld.lld
              chmod +x $out/bin/${stdenv.cc.targetPrefix}ld.lld
            '';
          };
        in
        pkgs.withCFlags [ "-g1" ] (stdenv.override (old: {
          allowedRequisites = null;
          cc = stdenv.cc.override { inherit bintools; };
        }));
      withReproducerCollectingStdenv = pkg: pkg.override {
        stdenv = reproducerCollectingStdenv pkgs.stdenv pkgs.buildPackages.lld;
      };
      withReproducerCollectingClangStdenv = pkg: pkg.override {
        clangStdenv = reproducerCollectingStdenv pkgs.clangStdenv pkgs.buildPackages.lld;
      };
    in
    {
      # For benchmarking the linker we want to disable LTO as otherwise we would
      # just be benchmarking the LLVM optimizer. Also, we generally want the
      # package to use the regular stdenv in order to simplify wrapping it.
      # Firefox normally uses the rustc stdenv but uses the regular one if
      # LTO is disabled so we kill two birds with one stone by disabling it.
      # Chromium uses the rustc stdenv unconditionally so we need the stuff
      # below to make sure that it finds our wrapped stdenv.
      chrome =
        (pkgs.chromium.override {
          newScope =
            extra:
            pkgs.newScope (
              extra
              // {
                pkgsBuildBuild = {
                  pkg-config = pkgs.pkgsBuildBuild.pkg-config;
                  rustc = {
                    llvmPackages = rec {
                      stdenv = reproducerCollectingStdenv pkgs.pkgsBuildBuild.rustc.llvmPackages.stdenv pkgs.pkgsBuildBuild.rustc.llvmPackages.lld;
                      bintools = stdenv.cc.bintools;
                    };
                  };
                };
              }
            );
          pkgs = {
            rustc = {
              llvmPackages = {
                stdenv = reproducerCollectingStdenv pkgs.rustc.llvmPackages.stdenv pkgs.rustc.llvmPackages.lld;
              };
            };
          };
        }).browser.overrideAttrs
          (old: {
            configurePhase =
              old.configurePhase
              + ''
                echo use_thin_lto = false >> out/Release/args.gn
                echo is_cfi = false >> out/Release/args.gn
              '';
          });
      firefox = (withReproducerCollectingStdenv pkgs.firefox-unwrapped).override {
        ltoSupport = false;
        pgoSupport = false;
      };
      # Won't work until https://github.com/NixOS/nixpkgs/pull/390631 lands.
      # Can replace above line with
      #   nixpkgsDir ? fetchTarball "https://github.com/NixOS/nixpkgs/archive/fbc5923fb30c7e1957a729f19f22968083fb473f.tar.gz",
      # for testing with that PR.
      linux-kernel = (withReproducerCollectingStdenv pkgs.linux_latest).dev;
      ladybird = withReproducerCollectingStdenv pkgs.ladybird;
      llvm = withReproducerCollectingStdenv pkgs.llvm;
      webkitgtk = withReproducerCollectingClangStdenv pkgs.webkitgtk;
      hello = withReproducerCollectingStdenv pkgs.hello;
    };
    targets = {
      x86_64 = reproducerPkgs { config = "x86_64-unknown-linux-gnu"; };
      aarch64 = reproducerPkgs { config = "aarch64-unknown-linux-gnu"; };
      riscv64 = reproducerPkgs { config = "riscv64-unknown-linux-gnu"; };
    };
    nativePkgs = nixpkgs { };
in
derivation {
  name = "lld-speed-test";
  system = builtins.currentSystem;
  builder = "${nativePkgs.bash}/bin/bash";
  args = [
    "-c"
    ''
      extract_reproducer() {
        ${nativePkgs.coreutils}/bin/mkdir -p $out/$2
        ${nativePkgs.llvm}/bin/llvm-objcopy -O binary --only-section=.lld_repro --set-section-flags .lld_repro=alloc $1 - | ${nativePkgs.gnutar}/bin/tar x -I ${nativePkgs.lib.getBin nativePkgs.buildPackages.lz4}/bin/lz4 --strip-components=1 -C $out/$2
      }

      extract_reproducer ${targets.aarch64.hello}/bin/hello hello-arm64
      extract_reproducer ${targets.x86_64.hello}/bin/hello hello-x64
      extract_reproducer ${targets.aarch64.chrome}/libexec/chromium/chromium chrome
      extract_reproducer ${targets.aarch64.ladybird}/lib/liblagom-web.so ladybird
      extract_reproducer ${targets.aarch64.firefox}/lib/firefox/libxul.so firefox-arm64
      extract_reproducer ${targets.x86_64.firefox}/lib/firefox/libxul.so firefox-x64
      extract_reproducer ${targets.riscv64.firefox}/lib/firefox/libxul.so firefox-riscv64
      extract_reproducer ${nativePkgs.lib.getLib targets.aarch64.llvm}/lib/libLLVM.so llvm-arm64
      extract_reproducer ${nativePkgs.lib.getLib targets.x86_64.llvm}/lib/libLLVM.so llvm-x64
      extract_reproducer ${nativePkgs.lib.getLib targets.riscv64.llvm}/lib/libLLVM.so llvm-riscv64
    ''
  ];
}
