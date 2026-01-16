# Build clang and compiler-rt targeting Baremetal RISC-V (cross-compiling)
#
# Use this as:
#
#  $ cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS=clang \
#    -DLLVM_ENABLE_RUNTIMES=compiler-rt \
#    -C../clang/cmake/caches/BaremetalRISCV.cmake ../llvm
#  $ ninja runtimes-riscv64-unknown-elf runtimes-riscv32-unknown-elf
#
set(TRIPLES riscv64-unknown-elf;riscv32-unknown-elf)
set(LLVM_BUILTIN_TARGETS ${TRIPLES} CACHE STRING "")
set(LLVM_RUNTIME_TARGETS ${TRIPLES} CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")

foreach(target ${TRIPLES})
  # builtins config
  set(BUILTINS_${target}_CMAKE_BUILD_TYPE Release CACHE STRING "")
  set(BUILTINS_${target}_CMAKE_SYSTEM_NAME Generic CACHE STRING "")
  set(BUILTINS_${target}_COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
  set(BUILTINS_${target}_COMPILER_RT_OS_DIR ${target} CACHE STRING "")
  set(BUILTINS_${target}_LLVM_CONFIG_NO_EXPORTS ON CACHE BOOL "")

  # enable compiler-rt in runtimes
  set(RUNTIMES_${target}_ENABLE_RUNTIMES compiler-rt CACHE STRING "")

  # runtimes config
  set(RUNTIMES_${target}_CMAKE_BUILD_TYPE Release CACHE STRING "")
  set(RUNTIMES_${target}_CMAKE_SYSTEM_NAME Generic CACHE STRING "")
  set(RUNTIMES_${target}_COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_OS_DIR ${target} CACHE STRING "")
  set(RUNTIMES_${target}_LLVM_CONFIG_NO_EXPORTS ON CACHE BOOL "")

  # additional runtimes config for compiler-rt
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_LIBFUZZER OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_PROFILE OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_SANITIZERS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_XRAY OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_USE_BUILTINS_LIBRARY ON CACHE BOOL "")
endforeach()
