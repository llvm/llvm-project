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
  nixpkgs ? fetchTarball "https://github.com/NixOS/nixpkgs/archive/992f916556fcfaa94451ebc7fc6e396134bbf5b1.tar.gz",
  system ? builtins.currentSystem,
}:
let
  reproducerPkgs =
    crossSystem:
    let
      pkgsCross = import nixpkgs { inherit crossSystem; };
      # Wraps the given stdenv and lld package into a variant that collects
      # the reproducer and builds with debug info.
      reproducerCollectingStdenv =
        stdenv: lld:
        let
          bintools = stdenv.cc.bintools.override {
            extraBuildCommands = ''
              wrap ${stdenv.cc.targetPrefix}nix-wrap-lld ${pkgsCross.path}/pkgs/build-support/bintools-wrapper/ld-wrapper.sh ${lld}/bin/ld.lld
              export lz4=${pkgsCross.lib.getBin pkgsCross.buildPackages.lz4}/bin/lz4
              substituteAll ${./ld-wrapper.sh} $out/bin/${stdenv.cc.targetPrefix}ld
              chmod +x $out/bin/${stdenv.cc.targetPrefix}ld
              substituteAll ${./ld-wrapper.sh} $out/bin/${stdenv.cc.targetPrefix}ld.lld
              chmod +x $out/bin/${stdenv.cc.targetPrefix}ld.lld
            '';
          };
        in
        pkgsCross.withCFlags [ "-g1" ] (
          stdenv.override (old: {
            allowedRequisites = null;
            cc = stdenv.cc.override { inherit bintools; };
          })
        );
      withReproducerCollectingStdenv =
        pkg:
        pkg.override {
          stdenv = reproducerCollectingStdenv pkgsCross.stdenv pkgsCross.buildPackages.lld;
        };
      withReproducerCollectingClangStdenv =
        pkg:
        pkg.override {
          clangStdenv = reproducerCollectingStdenv pkgsCross.clangStdenv pkgsCross.buildPackages.lld;
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
        (pkgsCross.chromium.override {
          newScope =
            extra:
            pkgsCross.newScope (
              extra
              // {
                pkgsBuildBuild = {
                  pkg-config = pkgsCross.pkgsBuildBuild.pkg-config;
                  rustc = {
                    llvmPackages = rec {
                      stdenv = reproducerCollectingStdenv pkgsCross.pkgsBuildBuild.rustc.llvmPackages.stdenv pkgsCross.pkgsBuildBuild.rustc.llvmPackages.lld;
                      bintools = stdenv.cc.bintools;
                    };
                  };
                };
              }
            );
          pkgs = {
            rustc = {
              llvmPackages = {
                stdenv = reproducerCollectingStdenv pkgsCross.rustc.llvmPackages.stdenv pkgsCross.rustc.llvmPackages.lld;
              };
            };
          };
        }).browser.overrideAttrs
          (old: {
            configurePhase = old.configurePhase + ''
              echo use_thin_lto = false >> out/Release/args.gn
              echo is_cfi = false >> out/Release/args.gn
            '';
          });
      firefox = (withReproducerCollectingStdenv pkgsCross.firefox-unwrapped).override {
        ltoSupport = false;
        pgoSupport = false;
      };
      # Doesn't work as-is because the kernel derivation calls the linker
      # directly instead of the wrapper. See:
      # https://github.com/NixOS/nixpkgs/blob/d3fdff1631946f3e51318317375d638dae3d6aa2/pkgs/os-specific/linux/kernel/common-flags.nix#L12
      linux-kernel = (withReproducerCollectingStdenv pkgsCross.linux_latest).dev;
      ladybird = withReproducerCollectingStdenv pkgsCross.ladybird;
      llvm = withReproducerCollectingStdenv pkgsCross.llvm;
      webkitgtk = withReproducerCollectingClangStdenv pkgsCross.webkitgtk;
      hello = withReproducerCollectingStdenv pkgsCross.hello;
    };
  targets = {
    x86_64 = reproducerPkgs { config = "x86_64-unknown-linux-gnu"; };
    aarch64 = reproducerPkgs { config = "aarch64-unknown-linux-gnu"; };
    riscv64 = reproducerPkgs { config = "riscv64-unknown-linux-gnu"; };
  };
  pkgs = import nixpkgs { };
in
pkgs.runCommand "lld-speed-test" { } ''
  extract_reproducer() {
    ${pkgs.coreutils}/bin/mkdir -p $out/$2
    ${pkgs.llvm}/bin/llvm-objcopy -O binary --only-section=.lld_repro --set-section-flags .lld_repro=alloc $1 - | ${pkgs.gnutar}/bin/tar x -I ${pkgs.lib.getBin pkgs.buildPackages.lz4}/bin/lz4 --strip-components=1 -C $out/$2
  }

  extract_reproducer ${targets.aarch64.hello}/bin/hello hello-arm64
  extract_reproducer ${targets.x86_64.hello}/bin/hello hello-x64
  extract_reproducer ${targets.aarch64.chrome}/libexec/chromium/chromium chrome
  extract_reproducer ${targets.aarch64.ladybird}/lib/liblagom-web.so ladybird
  extract_reproducer ${targets.aarch64.firefox}/lib/firefox/libxul.so firefox-arm64
  extract_reproducer ${targets.x86_64.firefox}/lib/firefox/libxul.so firefox-x64
  extract_reproducer ${targets.riscv64.firefox}/lib/firefox/libxul.so firefox-riscv64
  extract_reproducer ${pkgs.lib.getLib targets.aarch64.llvm}/lib/libLLVM.so llvm-arm64
  extract_reproducer ${pkgs.lib.getLib targets.x86_64.llvm}/lib/libLLVM.so llvm-x64
  extract_reproducer ${pkgs.lib.getLib targets.riscv64.llvm}/lib/libLLVM.so llvm-riscv6
''
