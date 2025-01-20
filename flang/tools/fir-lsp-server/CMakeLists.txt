set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

add_flang_tool(fir-lsp-server fir-lsp-server.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(fir-lsp-server PRIVATE
  CUFDialect
  FIRDialect
  HLFIRDialect
  MLIRLspServerLib
  ${dialect_libs}
  ${extension_libs})
