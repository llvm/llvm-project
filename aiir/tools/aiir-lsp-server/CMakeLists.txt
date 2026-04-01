set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

if(AIIR_INCLUDE_TESTS)
  set(test_libs
    AIIRAffineTransformsTestPasses
    AIIRShapeTestPasses
    AIIRSPIRVTestPasses
    AIIRTestAnalysis
    AIIRTestDialect
    AIIRTestDynDialect
    AIIRTestIR
    AIIRTestPass
    AIIRTestReducer
    )
  set(test_libs
    ${test_libs}
    AIIRTestTransformDialect
    AIIRTestTransforms)

  if (AIIR_ENABLE_PDL_IN_PATTERNMATCH)
    set(test_libs
      ${test_libs}
      AIIRTestRewrite)
  endif()
endif()

set(LIBS
  AIIRLspServerLib

  AIIRRegisterAllDialects
  AIIRRegisterAllExtensions
  AIIRRegisterAllPasses
  )

add_aiir_tool(aiir-lsp-server
  aiir-lsp-server.cpp
  )
aiir_target_link_libraries(aiir-lsp-server PRIVATE ${LIBS})
target_link_libraries(aiir-lsp-server PRIVATE ${test_libs})
llvm_update_compile_flags(aiir-lsp-server)

aiir_check_all_link_libraries(aiir-lsp-server)
