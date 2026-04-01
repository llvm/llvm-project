add_aiir_library(AIIROpenACCTestPasses
  TestOpenACC.cpp
  TestPointerLikeTypeInterface.cpp
  TestRecipePopulate.cpp
  TestOpenACCSupport.cpp
  
  EXCLUDE_FROM_LIBAIIR
)
aiir_target_link_libraries(AIIROpenACCTestPasses PUBLIC
  AIIRIR
  AIIRArithDialect
  AIIRFuncDialect
  AIIRMemRefDialect
  AIIROpenACCDialect
  AIIROpenACCAnalysis
  AIIRPass
  AIIRSupport
)

