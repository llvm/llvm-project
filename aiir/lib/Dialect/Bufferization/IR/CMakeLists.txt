add_aiir_dialect_library(AIIRBufferizationDialect
  AllocationOpInterface.cpp
  BufferizableOpInterface.cpp
  BufferDeallocationOpInterface.cpp
  BufferizationOps.cpp
  BufferizationDialect.cpp
  BufferViewFlowOpInterface.cpp
  UnstructuredControlFlow.cpp
  BufferizationTypeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Bufferization

  DEPENDS
  AIIRAllocationOpInterfaceIncGen
  AIIRBufferizationOpsIncGen
  AIIRBufferizationEnumsIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRDestinationStyleOpInterface
  AIIRDialect
  AIIRFuncDialect
  AIIRFunctionInterfaces
  AIIRIR
  AIIRSparseTensorDialect
  AIIRSubsetOpInterface
  AIIRTensorDialect
  AIIRMemRefDialect
  )
