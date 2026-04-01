add_aiir_library(AIIRLoopLikeInterfaceTestPasses
  TestBlockInLoop.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRLoopLikeInterfaceTestPasses PUBLIC
  AIIRPass
  AIIRLoopLikeInterface
  AIIRFuncDialect
  )
