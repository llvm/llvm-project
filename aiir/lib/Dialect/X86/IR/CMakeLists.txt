add_aiir_dialect_library(AIIRX86Dialect
  X86Dialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/X86

  DEPENDS
  AIIRX86IncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRSideEffectInterfaces
  )
