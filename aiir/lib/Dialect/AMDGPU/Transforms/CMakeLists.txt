add_aiir_dialect_library(AIIRAMDGPUTransforms
  EmulateAtomics.cpp
  FoldMemRefsOps.cpp
  MaskedloadToLoad.cpp
  ResolveStridedMetadata.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/AMDGPU/Transforms

  DEPENDS
  AIIRAMDGPUTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRAMDGPUDialect
  AIIRAMDGPUUtils
  AIIRAffineUtils
  AIIRArithDialect
  AIIRMemRefDialect
  AIIRSCFDialect
  AIIRVectorDialect
  AIIRControlFlowDialect
  AIIRFuncDialect
  AIIRIR
  AIIRPass
  AIIRTransforms
  AIIRTransformUtils
  )
