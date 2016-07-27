##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
)

clang_opencl_bc_builtins_lib(llvm ${CMAKE_CURRENT_SOURCE_DIR} ${sources})
