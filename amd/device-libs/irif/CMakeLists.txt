##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
)

set(irif_lib_file ${CMAKE_CURRENT_BINARY_DIR}/irif.amdgcn.bc)

add_custom_command(OUTPUT ${irif_lib_file}
  COMMAND ${LLVM_LINK} ${sources} -o ${irif_lib_file}
  DEPENDS ${sources})

add_custom_target(irif_lib DEPENDS ${irif_lib_file}
                           SOURCES ${sources})
set_target_properties(irif_lib PROPERTIES
  OUTPUT_NAME ${irif_lib_file}
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  ARCHIVE_OUTPUT_NAME "irif")

if(NOT ROCM_DEVICELIB_STANDALONE_BUILD)
  add_dependencies(irif_lib llvm-link)
endif()

# FIXME: Remove this once the usage in the runtime build is removed
list(APPEND AMDGCN_LIB_LIST irif_lib)
set(AMDGCN_LIB_LIST ${AMDGCN_LIB_LIST} PARENT_SCOPE)
