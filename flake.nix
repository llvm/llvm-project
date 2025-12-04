{
  description = "Flake to develop llvm toolchain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-circt.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nixpkgs-circt, flake-utils }:
    flake-utils.lib.eachDefaultSystem ( system:
    let
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      circt = (import nixpkgs-circt { inherit system; }).circt;
      pkgsRV = pkgs.pkgsCross.riscv64;
      targetLlvmLibraries = pkgsRV.llvmPackages_21;
      patched-libllvm = (targetLlvmLibraries.libllvm.override { src = ./.; });
    in rec {

      defaultPackage = packages.toolchain;

      packages.toolchain = with pkgsRV; (wrapCCWith rec {
        cc = (targetLlvmLibraries.clang-unwrapped.override {
                src = ./.;
                libllvm = patched-libllvm;
              });
        # libstdcxx is taken from gcc in an ad-hoc way in cc-wrapper.
        libcxx = null;
        extraPackages = [ targetLlvmLibraries.compiler-rt ];
        extraBuildCommands = ''
          rsrc="$out/resource-root"
          mkdir "$rsrc"
          ln -s "${lib.getLib cc}/lib/clang/*/include" "$rsrc"
          echo "-resource-dir=$rsrc" >> $out/nix-support/cc-cflags
          ln -s "${targetLlvmLibraries.compiler-rt.out}/lib" "$rsrc/lib"
          ln -s "${targetLlvmLibraries.compiler-rt.out}/share" "$rsrc/share"
        '';
      });

      packages.sim = pkgs.stdenv.mkDerivation {
        name = "fck-china-sim";
        # Floating derivation!!!
        __impure = true;

        src = pkgs.fetchgit {
          url = "https://github.com/OpenXiangShan/XiangShan.git";
          rev = "0fb84f8ddbfc9480d870f72cc903ac6453c888c9";
          fetchSubmodules = true;
          leaveDotGit = true;
          sha256 = "sha256-C+y//RJxI8FwYWCs8dmYLh8ZGVNCTAnRoiOVuY913Jg=";
          deepClone = false;
        };

        nativeBuildInputs = with pkgs; [
          mill
          time
          git
          espresso
          verilator
          python3
        ];

        buildInputs = with pkgs; [
          sqlite.dev
          zlib.dev
          zstd.dev
        ];

        buildPhase = ''
          runHook preBuild

          # Copy sources
          export NOOP_HOME=$out/src
          echo src = $src
          echo NOOP_HOME = $NOOP_HOME
          mkdir -p $NOOP_HOME
          cp -r $src/* $src/.* $NOOP_HOME

          # Patch shebangs
          chmod u+wx -R $NOOP_HOME
          patchShebangs --build $NOOP_HOME/scripts/

          # Build
          export _JAVA_OPTIONS="-XX:+UseZGC -XX:+ZUncommit -XX:ZUncommitDelay=30"
          FIRTOOL=${circt}/bin/firtool JVM_XMX=20G make -j8 -C $NOOP_HOME emu

          runHook postBuild
        '';

        installPhase = ''
          runHook preInstall

          mkdir -p $out/bin
          chmod u+x -R $out/src/build/
          cp $out/src/build/verilator-compile/emu $out/bin
          rm -rf $out/src

          runHook postInstall
        '';
      };

      devShells.default = targetLlvmLibraries.stdenv.mkDerivation {
        name = "devShell";
        # buildInputs = [];
        # ++ (with patched-libllvm; nativeBuildInputs ++ buildInputs ++ propagatedBuildInputs);

        cmakeFlags = [
          "-GNinja"
          "-DCMAKE_BUILD_TYPE=Debug"
          "-DLLVM_TARGETS_TO_BUILD=RISCV"
        ];

      };

    });
}

