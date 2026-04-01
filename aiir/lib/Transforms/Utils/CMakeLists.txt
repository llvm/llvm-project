add_aiir_library(AIIRTransformUtils
  CFGToSCF.cpp
  CommutativityUtils.cpp
  ControlFlowSinkUtils.cpp
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  Inliner.cpp
  InliningUtils.cpp
  LoopInvariantCodeMotionUtils.cpp
  RegionUtils.cpp
  WalkPatternRewriteDriver.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Transforms

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
  AIIRLoopLikeInterface
  AIIRSideEffectInterfaces
  AIIRSubsetOpInterface
  AIIRRewrite
  )
