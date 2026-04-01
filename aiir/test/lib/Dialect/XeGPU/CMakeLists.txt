add_aiir_dialect_library(AIIRXeGPUTestPasses
  TestXeGPUTransforms.cpp

  EXCLUDE_FROM_LIBAIIR
)

aiir_target_link_libraries(AIIRXeGPUTestPasses PUBLIC
  AIIRAffineUtils
  AIIRIR
  AIIRMemRefDialect
  AIIRXeGPUDialect
  AIIRPass
  AIIRTransforms
  AIIRGPUDialect
  AIIRXeGPUTransforms
)
