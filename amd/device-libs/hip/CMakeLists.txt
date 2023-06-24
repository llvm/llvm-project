##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB cl_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
)

file(GLOB ll_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
)

file(GLOB sources ${cl_sources} ${ll_sources})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ocml/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ockl/inc)
opencl_bc_lib(NAME hip
              SOURCES ${sources})
