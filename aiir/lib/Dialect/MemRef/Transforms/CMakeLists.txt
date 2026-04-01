add_aiir_dialect_library(AIIRMemRefTransforms
  AllocationOpInterfaceImpl.cpp
  BufferViewFlowOpInterfaceImpl.cpp
  ComposeSubView.cpp
  ElideReinterpretCast.cpp
  ExpandOps.cpp
  ExpandRealloc.cpp
  ExpandStridedMetadata.cpp
  EmulateWideInt.cpp
  EmulateNarrowType.cpp
  ExtractAddressComputations.cpp
  FlattenMemRefs.cpp
  FoldMemRefAliasOps.cpp
  IndependenceTransforms.cpp
  MultiBuffer.cpp
  NormalizeMemRefs.cpp
  ReifyResultShapes.cpp
  ResolveShapedTypeResultDims.cpp
  RuntimeOpVerification.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/MemRef

  DEPENDS
  AIIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  AIIRAffineTransforms
  AIIRAffineUtils
  AIIRArithDialect
  AIIRArithTransforms
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRGPUDialect
  AIIRInferTypeOpInterface
  AIIRLoopLikeInterface
  AIIRMemRefDialect
  AIIRMemRefUtils
  AIIRNVGPUDialect
  AIIRPass
  AIIRTensorDialect
  AIIRTransforms
  AIIRValueBoundsOpInterface
  AIIRVectorDialect
)

