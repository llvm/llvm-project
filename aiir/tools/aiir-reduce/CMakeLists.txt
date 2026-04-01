if(AIIR_INCLUDE_TESTS)
  set(test_libs
    AIIRTestDialect
    )
endif()

set(LIBS
  AIIRReduceLib
  AIIRRegisterAllDialects
  AIIRRegisterAllPasses
  )

add_aiir_tool(aiir-reduce
  aiir-reduce.cpp
  )

aiir_target_link_libraries(aiir-reduce PRIVATE ${LIBS})
target_link_libraries(aiir-reduce PRIVATE ${test_libs})
llvm_update_compile_flags(aiir-reduce)

aiir_check_all_link_libraries(aiir-reduce)
