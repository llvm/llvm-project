set(CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")

foreach(lang C;CXX;ASM)
    set(CMAKE_${lang}_FLAGS "-mfloat-abi=softfp -march=armv8m.main+fp+dsp -mcpu=cortex-m33" CACHE STRING "")
endforeach()

include(${CMAKE_CURRENT_LIST_DIR}/baremetal_common.cmake)
