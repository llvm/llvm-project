##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

clang_opencl_bc_builtins_lib(oclc ${CMAKE_CURRENT_SOURCE_DIR} ${sources})
