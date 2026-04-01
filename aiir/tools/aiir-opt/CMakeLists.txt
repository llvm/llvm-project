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
    ${cuda_test_libs}
    AIIRTestFuncToLLVM
    AIIRAffineTransformsTestPasses
    AIIRArithTestPasses
    AIIRArmSMETestPasses
    AIIRBufferizationTestPasses
    AIIRControlFlowTestPasses
    AIIRDLTITestPasses
    AIIRFuncTestPasses
    AIIRGPUTestPasses
    AIIRLinalgTestPasses
    AIIRLoopLikeInterfaceTestPasses
    AIIRMathTestPasses
    AIIRTestMathToVCIX
    AIIRMemRefTestPasses
    AIIRTestMemRefToLLVMWithTransforms
    AIIRShardTest
    AIIRNVGPUTestPasses
    AIIROpenACCTestPasses
    AIIRSCFTestPasses
    AIIRShapeTestPasses
    AIIRSPIRVTestPasses
    AIIRTensorTestPasses
    AIIRTestAnalysis
    AIIRTestConvertToSPIRV
    AIIRTestDialect
    AIIRTestDynDialect
    AIIRTestIR
    AIIRTestPass
    AIIRTestReducer
    AIIRTestTransforms
    AIIRTilingInterfaceTestPasses
    AIIRTosaTestPasses
    AIIRVectorTestPasses
    AIIRXeGPUTestPasses
    AIIRTestVectorToSPIRV
    AIIRLLVMTestPasses
    )
  set(test_libs ${test_libs}
    AIIRTestPDLL
    AIIRTestTransformDialect
    AIIRTestIRDLToCppDialect
    )

  if (AIIR_ENABLE_PDL_IN_PATTERNMATCH)
    set(test_libs ${test_libs}
      AIIRTestPDLL
      AIIRTestRewrite
      )
  endif()
endif()

set(LIBS

  AIIROptLib
  AIIRRegisterAllDialects
  AIIRRegisterAllExtensions
  AIIRRegisterAllPasses

  # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
  # registered directly in aiir-opt.cpp.
  AIIRToLLVMIRTranslationRegistration
  )

# Exclude from libAIIR.so because this has static options intended for
# opt-like tools only.
add_aiir_library(AIIRAiirOptMain
  aiir-opt.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRAiirOptMain PUBLIC
  ${LIBS}
  ${test_libs}
  )

add_aiir_tool(aiir-opt
  aiir-opt.cpp

  SUPPORT_PLUGINS
  )
aiir_target_link_libraries(aiir-opt PRIVATE ${LIBS})
target_link_libraries(aiir-opt PRIVATE ${test_libs})
llvm_update_compile_flags(aiir-opt)

aiir_check_all_link_libraries(aiir-opt)
export_executable_symbols_for_plugins(aiir-opt)
