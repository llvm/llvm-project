{
  description = "llvm-project — Clang Static Analyzer dev shell (NixOS)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # CMake find_package links zlib/zstd/libstdc++ by absolute nix-store path
      # without embedding an rpath, so the freshly-built llvm-min-tblgen / clang
      # fail mid-build with "libz.so.1" / "libstdc++.so.6: cannot open shared
      # object file". Preload them via LD_LIBRARY_PATH. Note: stdenv.cc.cc.lib
      # (the default gcc stdenv) provides libstdc++.so.6 — clangStdenv's does not.
      runtimeLibs = with pkgs; [
        zlib
        zstd
        libxml2
        ncurses
        libffi
        libedit
        stdenv.cc.cc.lib
      ];
    in
    {
      # Build clang + clang-tools-extra (Clang Static Analyzer) with clangStdenv.
      devShells.${system}.default = (pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
        nativeBuildInputs = with pkgs; [
          cmake
          ninja
          python3
          git
          lld
          sccache
        ];
        buildInputs = runtimeLibs;

        shellHook = ''
          export LLVM_SRC="$PWD"
          export CSA_BUILD="''${CSA_BUILD:-$LLVM_SRC/build/csa}"
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:''${LD_LIBRARY_PATH:-}"

          csa-configure() {
            cmake -G Ninja -S "$LLVM_SRC/llvm" -B "$CSA_BUILD" \
              -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" \
              -DCMAKE_BUILD_TYPE=Release \
              -DLLVM_ENABLE_ASSERTIONS=ON \
              -DLLVM_TARGETS_TO_BUILD=X86 \
              -DLLVM_USE_LINKER=lld \
              -DLLVM_OPTIMIZED_TABLEGEN=ON \
              -DBUILD_SHARED_LIBS=ON \
              -DCMAKE_C_COMPILER_LAUNCHER=sccache \
              -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
              "$@"
          }

          csa-build() {
            if [ $# -eq 0 ]; then
              ninja -C "$CSA_BUILD" clang clang-tidy
            else
              ninja -C "$CSA_BUILD" "$@"
            fi
          }

          csa-check() {
            if [ $# -eq 0 ]; then
              ninja -C "$CSA_BUILD" check-clang-analysis
            else
              ninja -C "$CSA_BUILD" "$@"
            fi
          }

          export -f csa-configure csa-build csa-check

          echo "CSA dev shell — csa-configure | csa-build [targets] | csa-check [targets]"
          echo "  build dir: $CSA_BUILD"
        '';
      };
    };
}
