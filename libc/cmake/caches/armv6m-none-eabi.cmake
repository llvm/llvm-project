set(CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(RUNTIMES_TARGET_TRIPLE "armv6m-none-eabi" CACHE STRING "")

foreach(lang C;CXX;ASM)
    set(CMAKE_${lang}_FLAGS "-march=armv6m -mcpu=cortex-m0plus -mfloat-abi=soft -Wno-atomic-alignment" CACHE STRING "")
endforeach()

include(${CMAKE_CURRENT_LIST_DIR}/baremetal_common.cmake)
