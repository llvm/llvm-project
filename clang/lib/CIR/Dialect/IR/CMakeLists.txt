add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRDialectBytecode.cpp
  CIRMemorySlot.cpp
  CIROpsArith.cpp
  CIROpsBuiltins.cpp
  CIROpsClassMembers.cpp
  CIROpsComplex.cpp
  CIROpsControlFlow.cpp
  CIROpsCore.cpp
  CIROpsCoroutine.cpp
  CIROpsEH.cpp
  CIROpsMath.cpp
  CIROpsSymbols.cpp
  CIROpsVectorMatrix.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp
  CIROpenCLAttrs.cpp

  DEPENDS
  MLIRCIROpsIncGen
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
