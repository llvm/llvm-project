add_clang_library(CIROpenACCSupport
  CIROpenACCOpsInterfaces.cpp
  CIROpenACCTypeInterfaces.cpp
  CIROpenACCUtils.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenACCDialect
  )
