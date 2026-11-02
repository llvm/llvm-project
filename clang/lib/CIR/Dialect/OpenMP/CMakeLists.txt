add_clang_library(CIROpenMPSupport
  RegisterOpenMPExtensions.cpp

  DEPENDS
  MLIRCIROpsIncGen
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenMPDialect
  )
