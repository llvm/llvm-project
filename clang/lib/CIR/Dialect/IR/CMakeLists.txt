add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRDialectBytecode.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp
  CIROpenCLAttrs.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRDialectBytecodeIncGen
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
  MLIRPtrMemorySpaceInterfaces
  clangAST
  )
