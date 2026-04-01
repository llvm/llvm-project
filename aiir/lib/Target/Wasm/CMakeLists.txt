add_aiir_translation_library(AIIRTargetWasmImport
  TranslateRegistration.cpp
  TranslateFromWasm.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/Target/Wasm

  LINK_LIBS PUBLIC
  AIIRWasmSSADialect
  AIIRIR
  AIIRSupport
  AIIRTranslateLib
)
