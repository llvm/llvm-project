add_mlir_library(InterCompiler
  Compiler.cpp

  LINK_LIBS PUBLIC
  InterEmit
  InterTransforms
  InterXeMachine
  InterXW
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTargetLLVMIRImport
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRUBDialect
  MLIRVectorDialect
  MLIRXeGPUDialect
  MLIRXeVMDialect
)
