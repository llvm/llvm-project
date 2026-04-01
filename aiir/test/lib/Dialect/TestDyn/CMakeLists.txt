add_aiir_dialect_library(AIIRTestDynDialect
  TestDynDialect.cpp

  EXCLUDE_FROM_LIBAIIR
)
aiir_target_link_libraries(AIIRTestDynDialect PUBLIC
  AIIRIR
)
