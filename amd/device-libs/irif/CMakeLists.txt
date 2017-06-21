##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
)

if (GENERIC_IS_ZERO)
  # HCC: Move .ll files to build directory and
  # adopt generic addr space as addr space 0
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  foreach(f ${sources} )
    file(COPY ${f}
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/transformed_src)
  endforeach(f)

  # Perform transformation
  execute_process(COMMAND "${CMAKE_SOURCE_DIR}/utils/change-addr-space.sh" "${CMAKE_SOURCE_DIR}/utils"
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  file(GLOB srcs
    ${CMAKE_CURRENT_BINARY_DIR}/transformed_src/*.ll
  )
else (GENERIC_IS_ZERO)
  file(GLOB srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
  )
endif (GENERIC_IS_ZERO)

opencl_bc_lib(irif ${srcs})
