# Declare a function to generate ODS with aiir-linalg-ods-yaml-gen
function(add_linalg_ods_yaml_gen yaml_ast_file output_file)
  set(YAML_AST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${yaml_ast_file})
  set(GEN_ODS_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.td)
  set(GEN_CPP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.cpp.inc)
  set_source_files_properties(
    ${GEN_ODS_FILE}
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
    ${GEN_CPP_FILE}
    PROPERTIES GENERATED TRUE)
  add_custom_command(
    OUTPUT ${GEN_ODS_FILE} ${GEN_CPP_FILE}
    COMMAND ${AIIR_LINALG_ODS_YAML_GEN_EXE} ${YAML_AST_SOURCE} -o-ods-decl=${GEN_ODS_FILE} -o-impl=${GEN_CPP_FILE}
    MAIN_DEPENDENCY
    ${YAML_AST_SOURCE}
    DEPENDS
    ${AIIR_LINALG_ODS_YAML_GEN_TARGET})
  add_custom_target(
    AIIR${output_file}YamlIncGen
    DEPENDS
    ${GEN_ODS_FILE} ${GEN_CPP_FILE})
  set_target_properties(AIIR${output_file}YamlIncGen PROPERTIES FOLDER "AIIR/Tablegenning")
  list(APPEND LLVM_TARGET_DEPENDS ${GEN_ODS_FILE})
  set(LLVM_TARGET_DEPENDS ${LLVM_TARGET_DEPENDS} PARENT_SCOPE)
endfunction()

# NOTE: `add_aiir_interface(interface)` adds `interface` as a dependency of
# aiir-generic-headers, i.e.:
#   * aiir-generic-headers -> interface
# In addition, we have an existing AIIR-wide dependency of:
#   * aiir-headers -> aiir-generic-headers.
# Now, observe that:
#   1. The targets below define _new_ dependencies for aiir-headers.
#   2. Before the new targets are defined, `add_linalg_ods_yaml_gen` updates
#     LLVM_TARGET_DEPENDS.
#   3. All tablegen targets pick-up LLVM_TARGET_DEPENDS.
# In order to avoid cyclic dependencies, we need to invoke
# `add_aiir_interface` (and update `aiir-generic-headers`) _before_
# LLVM_TARGET_DEPENDS is updated and new dependencies for `aiir-headers` are
# defined + added.
add_aiir_interface(RelayoutOpInterface)

# NOTE: LLVM_TARGET_DEPENDS gets picked up by tablegen targets to add file
# level dependencies. This is gross but CMake requires depending on both
# targets and generated files, and it must be done when the custom target is
# declared (there is no way to add after the fact).
set(LLVM_TARGET_DEPENDS "")
add_linalg_ods_yaml_gen(LinalgNamedStructuredOps.yaml LinalgNamedStructuredOps)

# Provide a short name for all external dependency that needs to
# include Linalg in ODS
add_custom_target(LinalgOdsGen
  DEPENDS
  AIIRLinalgNamedStructuredOpsYamlIncGen
)
set_target_properties(LinalgOdsGen PROPERTIES FOLDER "AIIR/Tablegenning")
add_dependencies(aiir-headers LinalgOdsGen)

add_aiir_dialect(LinalgOps linalg)

set(LLVM_TARGET_DEFINITIONS LinalgEnums.td)
aiir_tablegen(LinalgOpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(LinalgOpsEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgOps.td)
aiir_tablegen(LinalgOpsAttrDefs.h.inc -gen-attrdef-decls)
aiir_tablegen(LinalgOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgOpsAttributesIncGen)

add_aiir_doc(LinalgDoc LinalgOps Dialects/ -gen-op-doc -dialect=linalg)
add_dependencies(LinalgOpsDocGen LinalgOdsGen)

set(LLVM_TARGET_DEFINITIONS LinalgStructuredOps.td)
aiir_tablegen(LinalgStructuredOps.h.inc -gen-op-decls)
aiir_tablegen(LinalgStructuredOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgStructuredOpsIncGen)
add_dependencies(AIIRLinalgStructuredOpsIncGen LinalgOdsGen)

set(LLVM_TARGET_DEFINITIONS LinalgRelayoutOps.td)
aiir_tablegen(LinalgRelayoutOps.h.inc -gen-op-decls)
aiir_tablegen(LinalgRelayoutOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgRelayoutOpsIncGen)
add_dependencies(AIIRLinalgRelayoutOpsIncGen LinalgOdsGen)

set(LLVM_TARGET_DEFINITIONS LinalgInterfaces.td)
aiir_tablegen(LinalgInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(LinalgInterfaces.cpp.inc -gen-op-interface-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgInterfacesIncGen)

