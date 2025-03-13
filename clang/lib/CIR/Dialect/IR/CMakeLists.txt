add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIREnumsGen
  MLIRCIROpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIRInterfaces
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRCIRInterfaces
  clangAST
  )
