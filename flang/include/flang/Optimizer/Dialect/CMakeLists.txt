add_subdirectory(CUF)
add_subdirectory(FIRCG)
add_subdirectory(MIF)

# This replicates part of the add_aiir_dialect cmake function from AIIR that
# cannot be used her because it expects to be run inside AIIR directory which
# is not the case for FIR.
set(LLVM_TARGET_DEFINITIONS FIRDialect.td)
aiir_tablegen(FIRDialect.h.inc -gen-dialect-decls -dialect=fir)
aiir_tablegen(FIRDialect.cpp.inc -gen-dialect-defs -dialect=fir)

set(LLVM_TARGET_DEFINITIONS FIRAttr.td)
aiir_tablegen(FIREnumAttr.h.inc -gen-enum-decls)
aiir_tablegen(FIREnumAttr.cpp.inc -gen-enum-defs)
aiir_tablegen(FIRAttr.h.inc --gen-attrdef-decls)
aiir_tablegen(FIRAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS FIROps.td)
aiir_tablegen(FIROps.h.inc -gen-op-decls)
aiir_tablegen(FIROps.cpp.inc -gen-op-defs)
aiir_tablegen(FIROpsTypes.h.inc --gen-typedef-decls)
aiir_tablegen(FIROpsTypes.cpp.inc --gen-typedef-defs)
add_public_tablegen_target(FIROpsIncGen)

set(LLVM_TARGET_DEFINITIONS FortranVariableInterface.td)
aiir_tablegen(FortranVariableInterface.h.inc -gen-op-interface-decls)
aiir_tablegen(FortranVariableInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS FirAliasTagOpInterface.td)
aiir_tablegen(FirAliasTagOpInterface.h.inc -gen-op-interface-decls)
aiir_tablegen(FirAliasTagOpInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS SafeTempArrayCopyAttrInterface.td)
aiir_tablegen(SafeTempArrayCopyAttrInterface.h.inc -gen-attr-interface-decls)
aiir_tablegen(SafeTempArrayCopyAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(FIRSafeTempArrayCopyAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS FIROperationMoveOpInterface.td)
aiir_tablegen(FIROperationMoveOpInterface.h.inc -gen-op-interface-decls)
aiir_tablegen(FIROperationMoveOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(FIROperationMoveOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS CanonicalizationPatterns.td)
aiir_tablegen(CanonicalizationPatterns.inc -gen-rewriters)
add_public_tablegen_target(CanonicalizationPatternsIncGen)

add_custom_target(flang-doc)
set_target_properties(flang-doc PROPERTIES FOLDER "Flang/Docs")
set(dialect_doc_filename "FIRLangRef")

set(LLVM_TARGET_DEFINITIONS FIROps.td)
tablegen(AIIR ${dialect_doc_filename}.md -gen-op-doc -dialect=fir)
set(GEN_DOC_FILE ${FLANG_BINARY_DIR}/docs/Dialect/${dialect_doc_filename}.md)
add_custom_command(
        OUTPUT ${GEN_DOC_FILE}
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/${dialect_doc_filename}.md
                ${GEN_DOC_FILE}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${dialect_doc_filename}.md)
add_custom_target(${dialect_doc_filename}DocGen DEPENDS ${GEN_DOC_FILE})
set_target_properties(${dialect_doc_filename}DocGen PROPERTIES FOLDER "Flang/Tablegenning/Docs")
add_dependencies(flang-doc ${dialect_doc_filename}DocGen)
