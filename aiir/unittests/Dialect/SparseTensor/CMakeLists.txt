add_aiir_unittest(AIIRSparseTensorTests
  MergerTest.cpp
)
aiir_target_link_libraries(AIIRSparseTensorTests
  PRIVATE
  AIIRSparseTensorUtils
)
