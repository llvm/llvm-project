add_aiir_dialect_library(AIIRAffineUtils
  LoopFusionUtils.cpp
  LoopUtils.cpp
  Utils.cpp
  ViewLikeInterfaceUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Affine

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineAnalysis
  AIIRAnalysis
  AIIRArithUtils
  AIIRFuncDialect
  AIIRMemRefDialect
  AIIRTransformUtils
  AIIRViewLikeInterface
  )
