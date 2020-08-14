
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

  Transforms/ControlFlowConverter.cpp
  Transforms/CSE.cpp
  Transforms/Inliner.cpp
  Transforms/MemToReg.cpp
  Transforms/AffinePromotion.cpp
  Transforms/AffineDemotion.cpp
  Transforms/RewriteLoop.cpp
  Transforms/MemDataFlowOpt.cpp
  Transforms/FirLoopResultOpt.cpp

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
