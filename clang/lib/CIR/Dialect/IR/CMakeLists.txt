add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRAttrsEnumsGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRCIRInterfaces
  clangAST
  )
