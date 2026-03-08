#!/usr/bin/env bash

# Common flags for all systems
CMAKE_ARGS=(
    -G Ninja -S llvm -B build
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DLLVM_ENABLE_PROJECTS="clang;lld"
    -DLLVM_ENABLE_RUNTIMES="compiler-rt;libcxx;libcxxabi;libunwind"
    -DCLANG_DEFAULT_RTLIB=compiler-rt
    -DCLANG_DEFAULT_LINKER=lld
    -DLLVM_CCACHE_BUILD=ON
    -DLLVM_TARGETS_TO_BUILD=Native
    -DLLVM_OPTIMIZED_TABLEGEN=ON
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if [ "$(uname)" == "Darwin" ]; then # macOS-specific
    CMAKE_ARGS+=(
        -DDEFAULT_SYSROOT="$(xcrun --show-sdk-path)"
        -DCLANG_DEFAULT_CXX_STDLIB=libc++
    )
else # Ubuntu & compute cluster (Linux)
    CMAKE_ARGS+=(
        -DCLANG_DEFAULT_UNWINDLIB=libgcc
        -DCLANG_DEFAULT_CXX_STDLIB=libstdc++
        -DBUILD_SHARED_LIBS=ON
        -DLLVM_USE_SPLIT_DWARF=ON
    )
fi

echo "[+] Generating CMake configuration..."
cmake "${CMAKE_ARGS[@]}"
