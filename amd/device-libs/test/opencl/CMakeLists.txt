##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc)
clang_opencl_test(test_common ${CMAKE_CURRENT_SOURCE_DIR})
clang_opencl_test(test_math ${CMAKE_CURRENT_SOURCE_DIR})
clang_opencl_test(test_workitem ${CMAKE_CURRENT_SOURCE_DIR})
