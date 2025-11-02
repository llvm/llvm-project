add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp
  FPEnv.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRTypeConstraintsIncGen
  MLIRCIREnumsGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIRInterfaces
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRCIRInterfaces
  clangAST
  )
