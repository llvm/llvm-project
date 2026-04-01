if(AIIR_INCLUDE_TESTS)
  set(test_libs
    AIIRTestDialect
    )
endif()

add_aiir_tool(aiir-query
  aiir-query.cpp
  )
llvm_update_compile_flags(aiir-query)
aiir_target_link_libraries(aiir-query
  PRIVATE
  AIIRQueryLib
  AIIRRegisterAllDialects
  )
target_link_libraries(aiir-query PRIVATE ${test_libs})

aiir_check_link_libraries(aiir-query)
