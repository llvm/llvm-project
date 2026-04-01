add_aiir_translation_library(AIIRTargetCpp
  TranslateRegistration.cpp
  TranslateToCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  AIIRControlFlowDialect
  AIIREmitCDialect
  AIIRFuncDialect
  AIIRIR
  AIIRSupport
  AIIRTranslateLib
  )
