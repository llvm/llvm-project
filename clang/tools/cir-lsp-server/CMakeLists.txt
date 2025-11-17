include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

set(LIBS
  ${test_libs}
  clangCIR
  clangCIRLoweringDirectToLLVM
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCIR
  MLIRDialect
  MLIRIR
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRTransformUtils
  MLIRTransforms
  )

add_mlir_tool(cir-lsp-server
  cir-lsp-server.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(cir-lsp-server)
