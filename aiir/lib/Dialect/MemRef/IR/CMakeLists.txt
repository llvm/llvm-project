add_aiir_dialect_library(AIIRMemRefDialect
  MemRefDialect.cpp
  MemoryAccessOpInterfaces.cpp
  MemRefMemorySlot.cpp
  MemRefOps.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/aiir/Dialect/MemRef/IR

  DEPENDS
  AIIRMemRefOpsIncGen
  AIIRMemoryAccessOpInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRArithUtils
  AIIRControlFlowInterfaces
  AIIRDialect
  AIIRDialectUtils
  AIIRInferIntRangeCommon
  AIIRInferIntRangeInterface
  AIIRInferStridedMetadataInterface
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRMemOpInterfaces
  AIIRMemorySlotInterfaces
  AIIRShapedOpInterfaces
  AIIRSideEffectInterfaces
  AIIRUBDialect
  AIIRValueBoundsOpInterface
  AIIRViewLikeInterface
)
