# Cross-compilation additions: sysroot and clang symlinks — distribution build
#
# Loaded after hexagon-unknown-linux-musl-clang-dist.cmake to add
# cross-toolchain specifics.

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
  hexagon-h2-clang++
  hexagon-h2-clang
  hexagon-unknown-h2-clang++
  hexagon-unknown-h2-clang
  hexagon-unknown-h2-elf-clang++
  hexagon-unknown-h2-elf-clang
  hexagon-h2-elf-clang++
  hexagon-h2-elf-clang
  hexagon-qurt-clang++
  hexagon-qurt-clang
  hexagon-unknown-qurt-clang++
  hexagon-unknown-qurt-clang
  clang++
  clang-cl
  clang-cpp
  CACHE STRING "")
