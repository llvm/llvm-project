set(CMAKE_SYSTEM_PROCESSOR RISCV CACHE STRING "")
set(ARCH_TRIPLE "riscv32-unknown-elf" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/baremetal_common.cmake)
