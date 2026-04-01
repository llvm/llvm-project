set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY AIIR_TRANSLATION_LIBS)

add_llvm_executable(standalone-translate
  standalone-translate.cpp
  )
llvm_update_compile_flags(standalone-translate)
target_link_libraries(standalone-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  AIIRIR
  AIIRParser
  AIIRPass
  AIIRSPIRVDialect
  AIIRTranslateLib
  AIIRSupport
  )

aiir_check_link_libraries(standalone-translate)
