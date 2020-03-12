# This replicates part of the add_mlir_dialect cmake function from MLIR that
# cannot be used her because it expects to be run inside MLIR directory which
# is not the case for FIR.
set(LLVM_TARGET_DEFINITIONS FIROps.td)
mlir_tablegen(FIROps.h.inc -gen-op-decls)
mlir_tablegen(FIROps.cpp.inc -gen-op-defs)
add_public_tablegen_target(FIROpsIncGen)

add_custom_target(flang-doc)
set(dialect_doc_filename "FIRLangRef")

set(LLVM_TARGET_DEFINITIONS FIROps.td)
tablegen(MLIR ${dialect_doc_filename}.md -gen-op-doc "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
set(GEN_DOC_FILE ${FLANG_BINARY_DIR}/docs/Dialect/${dialect_doc_filename}.md)
add_custom_command(
        OUTPUT ${GEN_DOC_FILE}
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/${dialect_doc_filename}.md
                ${GEN_DOC_FILE}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${dialect_doc_filename}.md)
add_custom_target(${dialect_doc_filename}DocGen DEPENDS ${GEN_DOC_FILE})
add_dependencies(flang-doc ${dialect_doc_filename}DocGen)
