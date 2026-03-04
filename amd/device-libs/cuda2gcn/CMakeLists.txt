##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB cl_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
)

file(GLOB sources ${cl_sources})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ocml/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ockl/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
opencl_bc_lib(NAME cuda2gcn
              SOURCES ${sources})
