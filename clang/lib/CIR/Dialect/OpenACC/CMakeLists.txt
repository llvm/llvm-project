add_clang_library(CIROpenACCSupport
  CIROpenACCTypeInterfaces.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenACCDialect
  )
