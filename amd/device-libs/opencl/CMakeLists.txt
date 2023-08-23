##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB cl_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/async/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/devenq/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometric/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/image/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/integer/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/math/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/media/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/misc/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipes/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/relational/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/subgroup/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vldst/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/workgroup/*.cl
  )

file(GLOB sources ${cl_sources})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/integer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/workgroup)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ocml/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ockl/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../oclc/inc)

opencl_bc_lib(NAME opencl SOURCES ${sources})
