set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

add_flang_tool(fir-lsp-server fir-lsp-server.cpp)

get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY AIIR_EXTENSION_LIBS)
target_link_libraries(fir-lsp-server PRIVATE
  CUFDialect
  FIRDialect
  FIROpenACCSupport
  FIROpenMPSupport
  HLFIRDialect
  MIFDialect)

aiir_target_link_libraries(fir-lsp-server PRIVATE
  AIIRLspServerLib
  ${dialect_libs}
  ${extension_libs})
