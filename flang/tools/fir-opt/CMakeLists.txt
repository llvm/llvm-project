add_flang_tool(fir-opt fir-opt.cpp)
llvm_update_compile_flags(fir-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

if(FLANG_INCLUDE_TESTS)
  set(test_libs
    FIRTestAnalysis
    FIRTestOpenACCInterfaces
    MLIRTestIR
    )
endif()

target_link_libraries(fir-opt PRIVATE
  CUFAttrs
  CUFDialect
  FIRDialect
  FIRSupport
  FIRTransforms
  FIRCodeGen
  HLFIRDialect
  HLFIRTransforms
  FIROpenACCSupport
  FlangOpenMPTransforms
  FIRAnalysis
  ${test_libs}
)

mlir_target_link_libraries(fir-opt PRIVATE
  ${dialect_libs}
  ${extension_libs}

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
