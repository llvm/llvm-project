# FIXME: ideally we would just provide CURRENT_TOOLS_DIR globally instead of
#        computing it just for function(configure_lit_site_cfg)
set_llvm_build_mode()
string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" CURRENT_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")

add_custom_command(OUTPUT "${CURRENT_TOOLS_DIR}/hmaptool"
                   COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/hmaptool" "${CURRENT_TOOLS_DIR}"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hmaptool")

install(PROGRAMS hmaptool DESTINATION "${LLVM_TOOLS_INSTALL_DIR}" COMPONENT hmaptool)
add_custom_target(hmaptool ALL DEPENDS "${CURRENT_TOOLS_DIR}/hmaptool")
set_target_properties(hmaptool PROPERTIES FOLDER "Utils")

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-hmaptool"
                           DEPENDS hmaptool
                           COMPONENT hmaptool)
endif()
