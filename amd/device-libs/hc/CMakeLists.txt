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

if (GENERIC_IS_ZERO)
  # HCC: Move .ll files to build directory and
  # adopt generic addr space as addr space 0
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  foreach(f ${ll_sources} )
    file(COPY ${f}
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  endforeach(f)

  # Perform transformation
  execute_process(COMMAND "${CMAKE_SOURCE_DIR}/utils/change-addr-space.sh" "${CMAKE_SOURCE_DIR}/utils"
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  file(GLOB ll_srcs
    ${CMAKE_CURRENT_BINARY_DIR}/transformed_src/*.ll
  )
else (GENERIC_IS_ZERO)
  file(GLOB ll_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
  )
endif (GENERIC_IS_ZERO)

file(GLOB sources ${cl_sources} ${ll_srcs})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ocml/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ockl/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
opencl_bc_lib(hc ${sources})
