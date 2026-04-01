add_flang_tool(fir-opt fir-opt.cpp)
llvm_update_compile_flags(fir-opt)
get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY AIIR_EXTENSION_LIBS)

if(FLANG_INCLUDE_TESTS)
  set(test_libs
    FIRTestAnalysis
    FIRTestOpenACCInterfaces
    AIIROpenACCTestPasses
    AIIRTestIR
    )
endif()

target_link_libraries(fir-opt PRIVATE
  CUFAttrs
  CUFDialect
  FIRDialect
  FIRSupport
  FIRTransforms
  FIRCodeGen
  FIRCodeGenDialect
  HLFIRDialect
  HLFIRTransforms
  FIROpenACCSupport
  FIROpenACCTransforms
  FIROpenMPSupport
  FlangOpenMPTransforms
  FIRAnalysis
  MIFDialect
  ${test_libs}
)

aiir_target_link_libraries(fir-opt PRIVATE
  ${dialect_libs}
  ${extension_libs}

  # TODO: these should be transitive dependencies from a target providing
  # "registerFIRPasses()"
  AIIRIR
  AIIRLLVMDialect
  AIIRPass
  AIIRFuncToLLVM
  AIIRTransforms
  AIIRAffineToStandard
  AIIRAnalysis
  AIIRSCFToControlFlow
  AIIRParser
  AIIRSupport
  AIIRVectorToLLVM
  AIIROptLib
)
