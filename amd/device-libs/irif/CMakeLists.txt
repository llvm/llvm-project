##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
)

set(irif_lib_file ${CMAKE_CURRENT_BINARY_DIR}/irif.bc)

add_custom_command(OUTPUT ${irif_lib_file}
  COMMAND $<TARGET_FILE:llvm-link> ${sources} -o ${irif_lib_file}
  DEPENDS ${sources})

add_custom_target(irif DEPENDS ${irif_lib_file}
                       SOURCES ${sources})
set_target_properties(irif PROPERTIES
  OUTPUT_NAME ${irif_lib_file}
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  ARCHIVE_OUTPUT_NAME "irif")

if(NOT ROCM_DEVICELIB_STANDALONE_BUILD)
  add_dependencies(irif llvm-link)
endif()
