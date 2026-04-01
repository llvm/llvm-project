set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

set(LIBS
  AIIRIR
  AIIRPDLLAST
  AIIRPDLLCodeGen
  AIIRPDLLODS
  AIIRPDLLParser
  )

add_tablegen(aiir-pdll AIIR_PDLL
  DESTINATION "${AIIR_TOOLS_INSTALL_DIR}"
  EXPORT AIIR
  aiir-pdll.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(aiir-pdll PRIVATE ${LIBS})

aiir_check_all_link_libraries(aiir-pdll)
