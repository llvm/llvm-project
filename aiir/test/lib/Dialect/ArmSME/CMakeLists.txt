# Exclude tests from libAIIR.so
add_aiir_library(AIIRArmSMETestPasses
  TestLowerToArmSME.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRArmSMETestPasses PUBLIC
  AIIRArithToArmSME
  AIIRArmSMEToLLVM
  AIIRArmSMEToSCF
  AIIRArmSMETransforms
  AIIRArmSVETransforms
  AIIRIR
  AIIRPass
  AIIRTransforms
  AIIRVectorToArmSME
  AIIRVectorToSCF
  AIIRSCFToControlFlow
  )
