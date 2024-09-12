add_custom_target(clang-cir-doc)

# This replicates part of the add_mlir_doc cmake function from MLIR that cannot
# be used here. This happens because it expects to be run inside MLIR directory
# which is not the case for CIR (and also FIR, both have similar workarounds).
function(add_clang_mlir_doc doc_filename output_file output_directory command)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command} ${ARGN} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(GEN_DOC_FILE ${CLANG_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(clang-cir-doc ${output_file}DocGen)
endfunction()

add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CIR)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix CIR)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix CIR)
add_public_tablegen_target(MLIRCIRPassIncGen)

add_clang_mlir_doc(Passes CIRPasses ./ -gen-pass-doc)
