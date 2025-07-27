include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

set(LIBS
  ${test_libs}
  clangCIR
  clangCIRLoweringDirectToLLVM
  MLIRCIR
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRRegisterEverything
  )

add_mlir_tool(cir-lsp-server
  cir-lsp-server.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(cir-lsp-server)
