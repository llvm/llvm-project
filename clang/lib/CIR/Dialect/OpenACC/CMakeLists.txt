add_clang_library(CIROpenACCSupport
  CIROpenACCTypeInterfaces.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  MLIRCIRTypeConstraintsIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBytecodeOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenACCDialect
  )
