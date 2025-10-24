{
  description = "Flake to develop llvm toolchain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem ( system:
    let
      pkgs = import nixpkgs { inherit system; };
      gccForLibs = pkgs.stdenv.cc.cc;
    in rec {

      defaultPackage = pkgs.stdenv.mkDerivation {
        name = "llvm";

        dontUnpack = true;

        buildInputs = with pkgs; [
          python3
          ninja
          cmake
        ];

        cmakeFlags = [
            "-DUSE_DEPRECATED_GCC_INSTALL_PREFIX=1"
            "-DGCC_INSTALL_PREFIX=${pkgs.gcc}"
            "-DC_INCLUDE_DIRS=${pkgs.stdenv.cc.libc.dev}/include"
            "-GNinja"
            # Debug for debug builds
            "-DCMAKE_BUILD_TYPE=Release"
            # inst will be our installation prefix
            "-DCMAKE_INSTALL_PREFIX=../inst"
            "-DLLVM_INSTALL_TOOLCHAIN_ONLY=ON"
            # change this to enable the projects you need
            "-DLLVM_ENABLE_PROJECTS=clang"
            # enable libcxx* to come into play at runtimes
            "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi"
            # this makes llvm only to produce code for the current platform, this saves CPU time, change it to what you need
            "-DLLVM_TARGETS_TO_BUILD=host"
            "-S ${self}/llvm"
        ];
      };

      devShell = (defaultPackage.overrideAttrs (oldAttrs: {
        name = "llvm-env";
        buildInputs = oldAttrs.buildInputs ++ (with pkgs; [ verilator ]);
      }));

    });
}

