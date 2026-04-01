set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY AIIR_TRANSLATION_LIBS)

add_aiir_tool(aiir-translate
  aiir-translate.cpp
  )
llvm_update_compile_flags(aiir-translate)
aiir_target_link_libraries(aiir-translate
  PRIVATE
  ${dialect_libs}
  AIIRIR
  AIIRParser
  AIIRPass
  AIIRSPIRVDialect
  AIIRTranslateLib
  AIIRSupport
  )
target_link_libraries(aiir-translate
  PRIVATE
  ${translation_libs}
  ${test_libs}
  )

aiir_check_link_libraries(aiir-translate)
