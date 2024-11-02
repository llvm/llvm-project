add_flang_tool(fir-opt fir-opt.cpp)
llvm_update_compile_flags(fir-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

if(FLANG_INCLUDE_TESTS)
  set(test_libs
    FIRTestAnalysis
    )
endif()

target_link_libraries(fir-opt PRIVATE
  FIRDialect
  FIRSupport
  FIRTransforms
  FIRCodeGen
  HLFIRDialect
  FIRAnalysis
  ${test_libs}
  ${dialect_libs}

  # TODO: these should be transitive dependencies from a target providing
  # "registerFIRPasses()"
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRFuncToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToControlFlow
  MLIRParser
  MLIRSupport
  MLIRVectorToLLVM
  MLIROptLib

)
