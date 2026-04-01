add_aiir_dialect_library(AIIRXeGPUTransforms
  XeGPUBlocking.cpp
  XeGPUSgToWiDistributeExperimental.cpp
  XeGPUSubgroupDistribute.cpp
  XeGPUUnroll.cpp
  XeGPUWgToSgDistribute.cpp
  XeGPUPropagateLayout.cpp
  XeGPUVectorLinearize.cpp
  XeGPUPeepHoleOptimizer.cpp
  XeGPULayoutImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/XeGPU

  DEPENDS
  AIIRXeGPUPassIncGen

  LINK_LIBS PUBLIC
  AIIRAffineUtils
  AIIRIR
  AIIRIndexDialect
  AIIRMemRefDialect
  AIIRXeGPUDialect
  AIIRPass
  AIIRTransforms
  AIIRGPUDialect
  AIIRXeGPUUtils
  AIIRGPUUtils
  AIIRVectorTransforms
)
