##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../oclc/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(native_func_flags -fapprox-func)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/native_logF.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/native_expF.cl
  PROPERTIES COMPILE_FLAGS "${native_func_flags}")


# This implementation of sqrt will not be used through opencl, openmp,
# or hip. Compile to be correctly rounded just in case
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sqrtF.cl
  PROPERTIES COMPILE_FLAGS -cl-fp32-correctly-rounded-divide-sqrt)

opencl_bc_lib(NAME ocml SOURCES ${sources})
