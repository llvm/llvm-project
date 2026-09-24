set(CMAKE_SYSTEM_PROCESSOR hexagon CACHE STRING "")
set(RUNTIMES_TARGET_TRIPLE "hexagon-unknown-none-elf" CACHE STRING "")

# flag to build compiler-rt builtins
set(CMAKE_C_FLAGS "-mv68 -G0" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-mv68 -G0" CACHE STRING "")
set(CMAKE_ASM_FLAGS "-mv68 -G0" CACHE STRING "")

set(LLVM_ENABLE_RUNTIMES "libc;compiler-rt" CACHE STRING "")
set(LIBC_TEST_BUILTINS_TARGET "clang_rt.builtins-hexagon" CACHE STRING "")

set(LIBC_COMPILE_OPTIONS_DEFAULT
    "-mv68;-G0;-fuse-init-array"
    CACHE STRING "")
set(LIBC_LINK_OPTIONS_DEFAULT
    "-mv68;-G0"
    CACHE STRING "")
set(LIBC_TEST_COMPILE_OPTIONS_DEFAULT "-O0" CACHE STRING "")

# The bare-metal linker script requires explicit memory-region definitions for
# the test image.
set(LIBC_TEST_LINK_OPTIONS_DEFAULT
    "-mv68;-G0;-T;${CMAKE_CURRENT_LIST_DIR}/../../test/UnitTest/llvm-libc-baremetal.ld;-Wl,--defsym=__boot_flash=0x00100000;-Wl,--defsym=__boot_flash_size=0x00001000;-Wl,--defsym=__flash=0x00101000;-Wl,--defsym=__flash_size=0x003ff000;-Wl,--defsym=__ram=0x00500000;-Wl,--defsym=__ram_size=0x00800000;-Wl,--defsym=__stack_size=0x00040000"
    CACHE STRING "")
set(LIBC_TEST_CMD
    "qemu-system-hexagon -M sim -cpu v68 -kernel @BINARY@"
    CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/baremetal_common.cmake)

# Enable hermetic tests for the QEMU-backed configuration.
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "" FORCE)
