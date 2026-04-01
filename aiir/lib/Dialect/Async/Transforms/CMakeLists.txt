add_aiir_dialect_library(AIIRAsyncTransforms
  AsyncParallelFor.cpp
  AsyncRuntimeRefCounting.cpp
  AsyncRuntimeRefCountingOpt.cpp
  AsyncToAsyncRuntime.cpp
  PassDetail.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Async

  DEPENDS
  AIIRAsyncPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRAsyncDialect
  AIIRFuncDialect
  AIIRFunctionInterfaces
  AIIRIR
  AIIRPass
  AIIRSCFDialect
  AIIRSCFToControlFlow
  AIIRTransforms
  AIIRTransformUtils
)
