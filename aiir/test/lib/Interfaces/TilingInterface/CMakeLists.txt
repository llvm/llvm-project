set(LLVM_TARGET_DEFINITIONS TestTilingInterfaceTransformOps.td)
aiir_tablegen(TestTilingInterfaceTransformOps.h.inc -gen-op-decls)
aiir_tablegen(TestTilingInterfaceTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AIIRTestTilingInterfaceTransformOpsIncGen)

add_aiir_library(AIIRTilingInterfaceTestPasses
  TestTilingInterfaceTransformOps.cpp

  DEPENDS
  AIIRTestTilingInterfaceTransformOpsIncGen

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRTilingInterfaceTestPasses PUBLIC
  AIIRAffineDialect
  AIIRArithDialect
  AIIRIndexDialect
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRMemRefDialect
  AIIRSCFDialect
  AIIRSCFTransforms
  AIIRTensorDialect
  AIIRTensorTilingInterfaceImpl
  AIIRTransformDialect
  )
