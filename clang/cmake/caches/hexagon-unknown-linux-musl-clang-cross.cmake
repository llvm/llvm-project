# This file is for the llvm+clang options that are specific to building
# a cross-toolchain targeting hexagon linux.
set(DEFAULT_SYSROOT "../target/hexagon-unknown-linux-musl/" CACHE STRING "")
set(CLANG_LINKS_TO_CREATE
            hexagon-linux-musl-clang++
            hexagon-linux-musl-clang
            hexagon-unknown-linux-musl-clang++
            hexagon-unknown-linux-musl-clang
            hexagon-none-elf-clang++
            hexagon-none-elf-clang
            hexagon-unknown-none-elf-clang++
            hexagon-unknown-none-elf-clang
            clang++
            clang-cl
            clang-cpp
            CACHE STRING "")

# Note: FORCE is required to override the OFF setting in hexagon-unknown-linux-musl-clang.cmake
# which is loaded earlier in the -C chain.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "" FORCE)
