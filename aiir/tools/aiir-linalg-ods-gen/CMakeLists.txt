set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

# New aiir-linalg-ods-yaml-gen.
add_aiir_tool(aiir-linalg-ods-yaml-gen
  aiir-linalg-ods-yaml-gen.cpp
)
llvm_update_compile_flags(aiir-linalg-ods-yaml-gen)
target_link_libraries(aiir-linalg-ods-yaml-gen PRIVATE
  AIIRIR
  AIIRSupport
  AIIRParser
  )

setup_host_tool(aiir-linalg-ods-yaml-gen AIIR_LINALG_ODS_YAML_GEN AIIR_LINALG_ODS_YAML_GEN_EXE AIIR_LINALG_ODS_YAML_GEN_TARGET)

if(NOT ${AIIR_LINALG_ODS_YAML_GEN_EXE} STREQUAL "aiir-linalg-ods-yaml-gen")
  add_custom_target(aiir-linalg-ods-yaml-gen-host DEPENDS ${AIIR_LINALG_ODS_YAML_GEN_EXE})
  set_target_properties(aiir-linalg-ods-yaml-gen-host PROPERTIES FOLDER "AIIR/Tablegenning")

  if(NOT LLVM_BUILD_UTILS)
    set_target_properties(aiir-linalg-ods-yaml-gen PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()
endif()

configure_file(
  update_core_linalg_named_ops.sh.in
  ${AIIR_TOOLS_DIR}/update_core_linalg_named_ops.sh)
