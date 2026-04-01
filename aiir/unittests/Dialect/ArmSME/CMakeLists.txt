add_aiir_unittest(AIIRArmSMETests
  TileTypeConversionTest.cpp)
aiir_target_link_libraries(AIIRArmSMETests
  PRIVATE
  AIIRArmSMEToLLVM)
