add_clang_library(CIROpenMPSupport
  RegisterOpenMPExtensions.cpp

  DEPENDS
  MLIRCIROpsIncGen

  LINK_LIBS PUBLIC
  MLIRIRCore
  MLIRCIR
  MLIROpenMPDialect
  )
