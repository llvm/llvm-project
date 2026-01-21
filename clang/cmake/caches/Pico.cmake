# This file sets up a CMakeCache for the Raspberry Pi Pico toolchain build.

set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(LLVM_TARGETS_TO_BUILD ARM;RISCV CACHE STRING "")
set(LLVM_ENABLE_PROJECTS clang;lld;llvm CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES compiler-rt;libcxx;libc CACHE STRING "")

set(CLANG_DEFAULT_CXX_STDLIB libc++ CACHE STRING "")
set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB libunwind CACHE STRING "")

set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  llvm-ar
  llvm-cov
  llvm-objcopy
  llvm-objdump
  llvm-profdata
  llvm-ranlib
  llvm-readelf
  llvm-readobj
  llvm-size
  llvm-strings
  llvm-strip
  llvm-symbolizer
  CACHE STRING "")
set(LLVM_DISTRIBUTION_COMPONENTS
  builtins
  clang
  clang-resource-headers
  lld
  runtimes
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")

set(LLVM_BUILTIN_TARGETS armv6m-none-eabi;armv8m.main-none-eabi;riscv32-unknown-elf CACHE STRING "")
foreach(target ${LLVM_BUILTIN_TARGETS})
  set(BUILTINS_${target}_CMAKE_SYSTEM_NAME Generic CACHE STRING "")
  set(BUILTINS_${target}_CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "")
  set(BUILTINS_${target}_COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
endforeach()

set(BUILTINS_armv6m-none-eabi_CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(BUILTINS_armv8m.main-none-eabi_CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(BUILTINS_riscv32-unknown-elf_CMAKE_SYSTEM_PROCESSOR RISCV CACHE STRING "")
foreach(lang C;CXX;ASM)
  set(BUILTINS_armv6m-none-eabi_CMAKE_${lang}_FLAGS "-march=armv6m -mcpu=cortex-m0plus -mfloat-abi=soft" CACHE STRING "")
  set(BUILTINS_armv8m.main-none-eabi_CMAKE_${lang}_FLAGS "-march=armv8m.main+fp+dsp -mcpu=cortex-m33 -mfloat-abi=softfp" CACHE STRING "")
  set(BUILTINS_riscv32-unknown-elf_CMAKE_${lang}_FLAGS "-march=rv32imac_zicsr_zifencei_zba_zbb_zbs_zbkb -mabi=ilp32" CACHE STRING "")
endforeach()

set(LLVM_RUNTIME_TARGETS armv6m-none-eabi;armv8m.main-none-eabi;riscv32-unknown-elf CACHE STRING "")
foreach(target ${LLVM_RUNTIME_TARGETS})
  set(RUNTIMES_${target}_CMAKE_SYSTEM_NAME Generic CACHE STRING "")
  set(RUNTIMES_${target}_CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "")
  set(RUNTIMES_${target}_CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY CACHE STRING "")
  set(RUNTIMES_${target}_LLVM_ENABLE_RUNTIMES libc;libcxx CACHE STRING "")
  set(RUNTIMES_${target}_LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LLVM_LIBC_FULL_BUILD ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBC_ENABLE_USE_BY_CLANG ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_CXX_ABI none CACHE STRING "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_FILESYSTEM OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_RANDOM_DEVICE OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_LOCALIZATION OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_UNICODE OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_WIDE_CHARACTERS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_RTTI OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_THREADS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_MONOTONIC_CLOCK OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_SHARED_OUTPUT_NAME "c++-shared" CACHE STRING "")
  set(RUNTIMES_${target}_LIBCXX_HAS_PTHREAD_LIB OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_HAS_RT_LIB OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_HAS_TERMINAL_AVAILABLE OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
  set(RUNTIMES_${target}_RUNTIMES_USE_LIBC llvm-libc CACHE STRING "")
endforeach()

set(RUNTIMES_armv6m-none-eabi_CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(RUNTIMES_armv8m.main-none-eabi_CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(RUNTIMES_riscv32-unknown-elf_CMAKE_SYSTEM_PROCESSOR RISCV CACHE STRING "")
foreach(lang C;CXX;ASM)
  set(RUNTIMES_armv6m-none-eabi_CMAKE_${lang}_FLAGS "-march=armv6m -mcpu=cortex-m0plus -mfloat-abi=soft -Wno-atomic-alignment" CACHE STRING "")
  set(RUNTIMES_armv8m.main-none-eabi_CMAKE_${lang}_FLAGS "-march=armv8m.main+fp+dsp -mcpu=cortex-m33 -mfloat-abi=softfp -Wno-atomic-alignment" CACHE STRING "")
  set(RUNTIMES_riscv32-unknown-elf_CMAKE_${lang}_FLAGS "-march=rv32imac_zicsr_zifencei_zba_zbb_zbs_zbkb -mabi=ilp32 -Wno-atomic-alignment" CACHE STRING "")
endforeach()
