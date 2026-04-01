add_aiir_dialect_library(AIIRTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  ConcatOpPatterns.cpp
  EmptyOpPatterns.cpp
  ExtractSliceFromReshapeUtils.cpp
  FoldTensorSubsetOps.cpp
  IndependenceTransforms.cpp
  MergeConsecutiveInsertExtractSlicePatterns.cpp
  ReshapePatterns.cpp
  RewriteAsConstant.cpp
  RuntimeOpVerification.cpp
  SwapExtractSliceWithProducerPatterns.cpp
  SubsetInsertionOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tensor/Transforms

  DEPENDS
  AIIRTensorTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineTransforms
  AIIRAffineUtils
  AIIRArithDialect
  AIIRArithUtils
  AIIRDialectUtils
  AIIRIR
  AIIRLinalgDialect
  AIIRMemRefDialect
  AIIRPass
  AIIRRuntimeVerifiableOpInterface
  AIIRSCFDialect
  AIIRSubsetOpInterface
  AIIRTensorDialect
  AIIRTensorUtils
  AIIRTilingInterface
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorUtils
  AIIRValueBoundsOpInterface
)
