add_aiir_dialect_library(AIIRSparseTensorTransforms
  # Rewriting.
  BufferizableOpInterfaceImpl.cpp
  SparseAssembler.cpp
  SparseBufferRewriting.cpp
  SparseGPUCodegen.cpp
  SparseIterationToScf.cpp
  SparseReinterpretMap.cpp
  SparseStorageSpecifierToLLVM.cpp
  SparseSpaceCollapse.cpp
  SparseTensorCodegen.cpp
  SparseTensorConversion.cpp
  SparseTensorPasses.cpp
  SparseTensorRewriting.cpp
  SparseVectorization.cpp
  Sparsification.cpp
  SparsificationAndBufferizationPass.cpp
  StageSparseOperations.cpp
  # Utilities.
  Utils/CodegenEnv.cpp
  Utils/CodegenUtils.cpp
  Utils/IterationGraphSorter.cpp
  Utils/LoopEmitter.cpp
  Utils/SparseTensorDescriptor.cpp
  Utils/SparseTensorIterator.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SparseTensor

  DEPENDS
  AIIRSparseTensorPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRComplexDialect
  AIIRFuncDialect
  AIIRIR
  AIIRLLVMDialect
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRLLVMCommonConversion
  AIIRMemRefDialect
  AIIRPass
  AIIRSCFDialect
  AIIRSCFTransforms
  AIIRSCFUtils
  AIIRSparseTensorDialect
  AIIRSparseTensorEnums
  AIIRSparseTensorUtils
  AIIRTensorDialect
  AIIRTransforms
  AIIRVectorDialect
)
