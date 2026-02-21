# Build compiler-rt for Baremetal RISC-V (cross-compiling)
#
# Use this as:
#
#  $ cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS=clang \
#    -DLLVM_ENABLE_RUNTIMES=compiler-rt \
#    -C../clang/cmake/caches/BaremetalRISCV.cmake ../llvm
#  $ ninja runtimes-riscv64-unknown-elf runtimes-riscv32-unknown-elf
#
# Tested on Linux host; doesn't work on macOS host.
#
set(TRIPLES riscv64-unknown-elf;riscv32-unknown-elf)
set(LLVM_BUILTIN_TARGETS ${TRIPLES} CACHE STRING "")
set(LLVM_RUNTIME_TARGETS ${TRIPLES} CACHE STRING "")

foreach(target ${TRIPLES})
  # builtins config
  set(BUILTINS_${target}_CMAKE_BUILD_TYPE Release CACHE STRING "")
  set(BUILTINS_${target}_CMAKE_SYSTEM_NAME Generic CACHE STRING "")
  set(BUILTINS_${target}_COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
  set(BUILTINS_${target}_COMPILER_RT_OS_DIR ${target} CACHE STRING "")

  # enable compiler-rt in runtimes
  set(RUNTIMES_${target}_ENABLE_RUNTIMES compiler-rt CACHE STRING "")

  # runtimes config
  set(RUNTIMES_${target}_CMAKE_BUILD_TYPE Release CACHE STRING "")
  set(RUNTIMES_${target}_CMAKE_SYSTEM_NAME Generic CACHE STRING "")
  set(RUNTIMES_${target}_COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_OS_DIR ${target} CACHE STRING "")
endforeach()
