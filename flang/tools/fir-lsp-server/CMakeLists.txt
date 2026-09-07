set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

add_flang_tool(fir-lsp-server fir-lsp-server.cpp)

target_link_libraries(fir-lsp-server PRIVATE
  CUFDialect
  FIRDialect
  FIROpenACCSupport
  FIROpenMPSupport
  HLFIRDialect
  MIFDialect)

mlir_target_link_libraries(fir-lsp-server PRIVATE
  MLIRLspServerLib
  ${FLANG_MLIR_DIALECT_LIBS}
  ${FLANG_MLIR_EXTENSION_LIBS})
