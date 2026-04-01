set(LIBS
  AIIRPdllLspServerLib
  )

add_aiir_tool(aiir-pdll-lsp-server
  aiir-pdll-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(aiir-pdll-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(aiir-pdll-lsp-server)

aiir_check_all_link_libraries(aiir-pdll-lsp-server)
