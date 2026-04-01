add_aiir_unittest(AIIRAMDGPUTests
  AMDGPUUtilsTest.cpp
)
aiir_target_link_libraries(AIIRAMDGPUTests
  PRIVATE
  AIIRAMDGPUUtils
)
