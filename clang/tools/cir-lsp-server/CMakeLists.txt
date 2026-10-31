include_directories(${CLANG_CIR_MLIR_INCLUDE_DIRS})

set(LIBS
  ${test_libs}
  clangCIR
  CIRRegisterAllDialects
  clangCIRLoweringDirectToLLVM
  MLIRAnalysis
  MLIRCIR
  MLIRDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLspServerLib
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
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
