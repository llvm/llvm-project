add_clang_library(CIROpenACCSupport
  CIROpenACCTypeInterfaces.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  MLIRCIRTypeConstraintsIncGen
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenACCDialect
  )
