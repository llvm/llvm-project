add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRDialectBytecode.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp
  CIROpenCLAttrs.cpp

  DEPENDS
  MLIRCIRDialectBytecodeIncGen
  MLIRCIREnumsGen
  mlir-headers

  LINK_COMPONENTS
  TargetParser

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
