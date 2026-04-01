add_aiir_dialect_library(AIIRShapeOpsTransforms
  BufferizableOpInterfaceImpl.cpp
  OutlineShapeComputation.cpp
  RemoveShapeConstraints.cpp
  ShapeToShapeLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ShapeOps/Transforms

  DEPENDS
  AIIRShapeTransformsIncGen
  )

target_link_libraries(AIIRShapeOpsTransforms
  PUBLIC
  AIIRArithDialect
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRIR
  AIIRMemRefDialect
  AIIRPass
  AIIRShapeDialect
  AIIRSupport
  AIIRTransforms
  )
