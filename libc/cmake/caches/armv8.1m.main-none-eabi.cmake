set(CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(RUNTIMES_TARGET_TRIPLE "armv8.1m.main-none-eabi" CACHE STRING "")

foreach(lang C;CXX;ASM)
    set(CMAKE_${lang}_FLAGS "-mfloat-abi=hard -march=armv8.1-m.main+mve.fp+fp.dp -mcpu=cortex-m55" CACHE STRING "")
endforeach()

include(${CMAKE_CURRENT_LIST_DIR}/baremetal_common.cmake)
