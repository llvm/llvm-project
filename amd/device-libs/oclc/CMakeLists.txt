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

foreach (file ${sources})
  get_filename_component(dir ${file} DIRECTORY)
  get_filename_component(name ${file} NAME_WE)
  get_filename_component(ext ${file} EXT)
  opencl_bc_lib(oclc_${name} ${file})
endforeach()
