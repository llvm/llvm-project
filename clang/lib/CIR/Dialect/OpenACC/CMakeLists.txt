add_clang_library(CIROpenACCSupport
  CIROpenACCTypeInterfaces.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  MLIRCIRTypeConstraintsIncGen
  MLIRArithOpsIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBytecodeOpInterfaceIncGen
  MLIRCallInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenACCDialect
  )
