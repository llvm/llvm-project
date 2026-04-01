add_subdirectory(Utils)

add_aiir_library(AIIRTransforms
  Canonicalizer.cpp
  CompositePass.cpp
  ControlFlowSink.cpp
  CSE.cpp
  GenerateRuntimeVerification.cpp
  BubbleDownMemorySpaceCasts.cpp
  InlinerPass.cpp
  LocationSnapshot.cpp
  LoopInvariantCodeMotion.cpp
  Mem2Reg.cpp
  OpStats.cpp
  PrintIR.cpp
  RemoveDeadValues.cpp
  SCCP.cpp
  SROA.cpp
  StripDebugInfo.cpp
  SymbolDCE.cpp
  SymbolPrivatize.cpp
  TopologicalSort.cpp
  ViewOpGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Transforms

  DEPENDS
  AIIRTransformsPassIncGen
  AIIRTransformsDialectInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRFunctionInterfaces
  AIIRLoopLikeInterface
  AIIRMemOpInterfaces
  AIIRMemorySlotInterfaces
  AIIRPass
  AIIRRuntimeVerifiableOpInterface
  AIIRSideEffectInterfaces
  AIIRSupport
  AIIRTransformUtils
  AIIRUBDialect
  )
