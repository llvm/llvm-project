##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB cl_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/async/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometric/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/image/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/integer/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/math/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/media/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/misc/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/relational/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/subgroup/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vldst/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/workgroup/*.cl
)

file(GLOB ll_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/workgroup/*.ll
)

if (GENERIC_IS_ZERO)
  # HCC: Move .ll files to build directory and
  # adopt generic addr space as addr space 0
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/transformed_src/workgroup)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  foreach(f ${ll_sources} )
    file(COPY ${f}
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/transformed_src/workgroup)
  endforeach(f)

  # Perform transformation
  execute_process(COMMAND "${CMAKE_SOURCE_DIR}/utils/change-addr-space.sh" "${CMAKE_SOURCE_DIR}/utils"
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  file(GLOB ll_srcs
    ${CMAKE_CURRENT_BINARY_DIR}/transformed_src/workgroup/*.ll
  )
else (GENERIC_IS_ZERO)
  file(GLOB ll_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/workgroup/*.ll
  )

endif (GENERIC_IS_ZERO)

file(GLOB sources ${cl_sources} ${ll_srcs})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/integer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/workgroup)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ocml/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ockl/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../oclc/inc)
opencl_bc_lib(opencl ${sources})
