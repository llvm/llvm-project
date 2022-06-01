add_custom_command(OUTPUT "${LLVM_TOOLS_BINARY_DIR}/hmaptool"
                   COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/hmaptool" "${LLVM_TOOLS_BINARY_DIR}"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hmaptool")

install(PROGRAMS hmaptool DESTINATION "${LLVM_UTILS_INSTALL_DIR}" COMPONENT hmaptool)
add_custom_target(hmaptool ALL DEPENDS "${LLVM_TOOLS_BINARY_DIR}/hmaptool")
set_target_properties(hmaptool PROPERTIES FOLDER "Utils")

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-hmaptool"
                           DEPENDS hmaptool
                           COMPONENT hmaptool)
endif()
