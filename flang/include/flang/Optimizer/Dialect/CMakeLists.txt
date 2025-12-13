add_subdirectory(CUF)
add_subdirectory(FIRCG)
add_subdirectory(MIF)

# This replicates part of the add_mlir_dialect cmake function from MLIR that
# cannot be used her because it expects to be run inside MLIR directory which
# is not the case for FIR.
set(LLVM_TARGET_DEFINITIONS FIRDialect.td)
mlir_tablegen(FIRDialect.h.inc -gen-dialect-decls -dialect=fir)
mlir_tablegen(FIRDialect.cpp.inc -gen-dialect-defs -dialect=fir)

set(LLVM_TARGET_DEFINITIONS FIRAttr.td)
mlir_tablegen(FIREnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(FIREnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(FIRAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(FIRAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS FIROps.td)
mlir_tablegen(FIROps.h.inc -gen-op-decls)
mlir_tablegen(FIROps.cpp.inc -gen-op-defs)
mlir_tablegen(FIROpsTypes.h.inc --gen-typedef-decls)
mlir_tablegen(FIROpsTypes.cpp.inc --gen-typedef-defs)
add_public_tablegen_target(FIROpsIncGen)

set(LLVM_TARGET_DEFINITIONS FortranVariableInterface.td)
mlir_tablegen(FortranVariableInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(FortranVariableInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS FirAliasTagOpInterface.td)
mlir_tablegen(FirAliasTagOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(FirAliasTagOpInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS SafeTempArrayCopyAttrInterface.td)
mlir_tablegen(SafeTempArrayCopyAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(SafeTempArrayCopyAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(FIRSafeTempArrayCopyAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS CanonicalizationPatterns.td)
mlir_tablegen(CanonicalizationPatterns.inc -gen-rewriters)
add_public_tablegen_target(CanonicalizationPatternsIncGen)

add_custom_target(flang-doc)
set_target_properties(flang-doc PROPERTIES FOLDER "Flang/Docs")
set(dialect_doc_filename "FIRLangRef")

set(LLVM_TARGET_DEFINITIONS FIROps.td)
tablegen(MLIR ${dialect_doc_filename}.md -gen-op-doc)
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
