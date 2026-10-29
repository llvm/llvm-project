add_clang_library(CIROpenMPSupport
  RegisterOpenMPExtensions.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRMemorySlotInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIR
  MLIROpenMPDialect
  )
