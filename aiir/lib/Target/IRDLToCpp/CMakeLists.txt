add_aiir_translation_library(AIIRTargetIRDLToCpp
  TranslationRegistration.cpp
  IRDLToCpp.cpp
  Templates/DialectDecl.txt
  Templates/DialectDef.txt
  Templates/Header.txt
  Templates/PerOperationDecl.txt
  Templates/PerOperationDef.txt
  Templates/TypeDecl.txt
  Templates/TypeDef.txt
  Templates/TypeHeaderDecl.txt
  Templates/TypeHeaderDef.txt

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRIRDL
  AIIRTranslateLib
)
