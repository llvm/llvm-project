set(CMAKE_SYSTEM_PROCESSOR arm CACHE STRING "")
set(RUNTIMES_TARGET_TRIPLE "armv7m-none-eabi" CACHE STRING "")

foreach(lang C;CXX;ASM)
    set(CMAKE_${lang}_FLAGS "-march=armv7m -mcpu=cortex-m4 -mfloat-abi=soft -Wno-atomic-alignment \"-Dvfprintf(stream, format, vlist)=vprintf(format, vlist)\" \"-Dfprintf(stream, format, ...)=printf(format)\" \"-Dfputs(string, stream)=puts(string)\" -D_LIBCPP_PRINT=1" CACHE STRING "")
endforeach()

include(${CMAKE_CURRENT_LIST_DIR}/baremetal_common.cmake)
