
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIROptimizer
  Dialect/FIRAttr.cpp
  Dialect/FIRDialect.cpp
  Dialect/FIROps.cpp
  Dialect/FIRType.cpp

  Support/FIRContext.cpp
  Support/InternalNames.cpp
  Support/KindMapping.cpp

  Analysis/IteratedDominanceFrontier.cpp

  CodeGen/CodeGen.cpp
  CodeGen/PreCGRewrite.cpp
  CodeGen/Target.cpp
  CodeGen/TargetRewrite.cpp

  Transforms/AffineDemotion.cpp
  Transforms/AffinePromotion.cpp
  Transforms/ArrayValueCopy.cpp
  Transforms/ControlFlowConverter.cpp
  Transforms/CSE.cpp
  Transforms/FirLoopResultOpt.cpp
  Transforms/Inliner.cpp
  Transforms/MemDataFlowOpt.cpp
  Transforms/MemToReg.cpp
  Transforms/RewriteLoop.cpp

  DEPENDS
  FIROpsIncGen
  FIROptCodeGenPassIncGen
  FIROptTransformsPassIncGen
  RewritePatternsIncGen
  ${dialect_libs}

  LINK_LIBS
  ${dialect_libs}
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
