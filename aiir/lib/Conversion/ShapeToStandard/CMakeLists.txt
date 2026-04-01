set(LLVM_TARGET_DEFINITIONS ShapeToStandard.td)
aiir_tablegen(ShapeToStandard.cpp.inc -gen-rewriters)
add_public_tablegen_target(ShapeToStandardIncGen)

add_aiir_conversion_library(AIIRShapeToStandard
  ConvertShapeConstraints.cpp
  ShapeToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ShapeToStandard

  DEPENDS
  AIIRConversionPassIncGen
  ShapeToStandardIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRIR
  AIIRShapeDialect
  AIIRTensorDialect
  AIIRPass
  AIIRSCFDialect
  AIIRTransforms
  )
