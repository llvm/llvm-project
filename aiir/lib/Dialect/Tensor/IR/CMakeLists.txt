set(LLVM_OPTIONAL_SOURCES
  TensorDialect.cpp
  TensorInferTypeOpInterfaceImpl.cpp
  TensorOps.cpp
  TensorTilingInterfaceImpl.cpp
  ValueBoundsOpInterfaceImpl.cpp
)

add_aiir_dialect_library(AIIRTensorDialect
  TensorDialect.cpp
  TensorOps.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aiir/Dialect/Tensor

  DEPENDS
  AIIRTensorOpsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRArithUtils
  AIIRCastInterfaces
  AIIRComplexDialect
  AIIRDestinationStyleOpInterface
  AIIRDialectUtils
  AIIRIR
  AIIRInferIntRangeCommon
  AIIRInferIntRangeInterface
  AIIRInferTypeOpInterface
  AIIRParallelCombiningOpInterface
  AIIRRuntimeVerifiableOpInterface
  AIIRShapedOpInterfaces
  AIIRSideEffectInterfaces
  AIIRSupport
  AIIRValueBoundsOpInterface
  AIIRViewLikeInterface
  )

add_aiir_dialect_library(AIIRTensorInferTypeOpInterfaceImpl
  TensorInferTypeOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tensor

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRIR
  AIIRInferTypeOpInterface
  AIIRSupport
  AIIRTensorDialect
  )

add_aiir_dialect_library(AIIRTensorTilingInterfaceImpl
  TensorTilingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tensor

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineUtils
  AIIRArithUtils
  AIIRDialectUtils
  AIIRIR
  AIIRLinalgDialect
  AIIRLinalgUtils
  AIIRSCFDialect
  AIIRSupport
  AIIRTensorDialect
  AIIRTensorUtils
  AIIRTilingInterface
  AIIRValueBoundsOpInterface
  )
