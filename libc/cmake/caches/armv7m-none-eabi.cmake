set(CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(RUNTIMES_TARGET_TRIPLE "armv7m-none-eabi" CACHE STRING "")
set(LLVM_LIBC_FULL_BUILD ON CACHE BOOL "")

foreach(lang C;CXX;ASM)
  set(CMAKE_${lang}_FLAGS "-march=armv7-m -mcpu=cortex-m4 -mfloat-abi=soft -mfpu=none" CACHE STRING "")
endforeach()

# Offload the errno definition to libc
set(LIBC_CONF_ERRNO_MODE LIBC_ERRNO_MODE_SHARED CACHE STRING "")
# Enable floating-point conversions in printf. This became practical with the
# introduction of float320.
set(LIBC_CONF_PRINTF_DISABLE_FLOAT OFF CACHE BOOL "")
set(LIBC_CONF_PRINTF_FLOAT_TO_STR_USE_FLOAT320 ON CACHE BOOL "")
# Discard unused printf functionality
set(LIBC_CONF_PRINTF_MODULAR ON CACHE BOOL "")

# Configs for testing and applications are only done if compiler-rt is enabled,
# as the latter is a dependency for full programs.
if(compiler-rt IN_LIST LLVM_ENABLE_RUNTIMES)
  set(LLVM_INCLUDE_TESTS ON CACHE BOOL "")
  set(LIBC_TEST_BUILTINS_TARGET "clang_rt.builtins-armv7m" CACHE STRING "")

  # Memory mappings
  set(__boot_flash 0x00000000)
  set(__boot_flash_size 0x1000)
  set(__flash 0x21000000)
  set(__flash_size 0x600000)
  set(__ram 0x21600000)
  set(__ram_size 0xA00000)
  set(__stack_size 0x1000)
  set(SYMBOLS
    __boot_flash
    __boot_flash_size
    __flash
    __flash_size
    __ram
    __ram_size
    __stack_size
  )

  set(DEFSYM_ARGS "")
  foreach(symbol IN LISTS SYMBOLS)
    list(APPEND DEFSYM_ARGS "-Wl,--defsym=${symbol}=${${symbol}}")
  endforeach()

  set(LINKER_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../../test/UnitTest/llvm-libc-baremetal.ld")

  set(LIBC_TEST_LINK_OPTIONS_DEFAULT "-T;${LINKER_SCRIPT};${DEFSYM_ARGS}" CACHE STRING "")

  set(LIBC_TEST_CMD "qemu-system-arm -M mps2-an386 -cpu cortex-m4 -chardev stdio,mux=on,id=stdio0 -semihosting-config enable=on,chardev=stdio0 -monitor none -serial none -nographic -kernel @BINARY@" CACHE STRING "")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/baremetal_common.cmake)
