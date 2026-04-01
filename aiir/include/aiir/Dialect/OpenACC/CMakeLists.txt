add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OpenACC/ACC.td)
aiir_tablegen(AccCommon.td --gen-directive-decl --directives-dialect=OpenACC)
add_aiir_dialect_tablegen_target(acc_common_td)

add_aiir_dialect(OpenACCOps acc)

add_aiir_doc(OpenACCOps OpenACCDialectOps Dialects/ -gen-dialect-doc -dialect=acc)
add_dependencies(OpenACCDialectOpsDocGen acc_common_td)

set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
aiir_tablegen(OpenACCOpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(OpenACCOpsEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIROpenACCEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
aiir_tablegen(OpenACCOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=acc)
aiir_tablegen(OpenACCOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=acc)
add_aiir_dialect_tablegen_target(AIIROpenACCAttributesIncGen)

add_aiir_interface(OpenACCOpsInterfaces)

set(LLVM_TARGET_DEFINITIONS OpenACCTypeInterfaces.td)
aiir_tablegen(OpenACCTypeInterfaces.h.inc -gen-type-interface-decls)
aiir_tablegen(OpenACCTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_aiir_dialect_tablegen_target(AIIROpenACCTypeInterfacesIncGen)
