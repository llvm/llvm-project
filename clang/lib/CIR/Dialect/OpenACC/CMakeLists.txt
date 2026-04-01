add_clang_library(CIROpenACCSupport
  CIROpenACCTypeInterfaces.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  AIIRCIRTypeConstraintsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRCIR
  AIIROpenACCDialect
  )
