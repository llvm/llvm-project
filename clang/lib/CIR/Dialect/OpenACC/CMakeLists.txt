add_clang_library(CIROpenACCSupport
  CIROpenACCTypeInterfaces.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  MLIRCIRTypeConstraintsIncGen
  MLIRArithOpsIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBuiltinTypeInterfacesIncGen
  MLIRBytecodeOpInterfaceIncGen
  MLIRCallInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenACCDialect
  )
