add_aiir_dialect_library(AIIRSCFDialect
  SCF.cpp
  DeviceMappingInterface.cpp
  MemorySlot.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/SCF

  DEPENDS
  AIIRSCFOpsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRControlFlowDialect
  AIIRDialectUtils
  AIIRFunctionInterfaces
  AIIRIR
  AIIRLoopLikeInterface
  AIIRMemorySlotUtils
  AIIRSideEffectInterfaces
  AIIRTensorDialect
  AIIRValueBoundsOpInterface
  AIIRTransformUtils
  )
