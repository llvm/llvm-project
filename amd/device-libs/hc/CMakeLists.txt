##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ocml/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ockl/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
clang_opencl_bc_builtins_lib(hc ${CMAKE_CURRENT_SOURCE_DIR} ${sources})
