add_flang_tool(fir-opt fir-opt.cpp)
llvm_update_compile_flags(fir-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_library(fir-opt-common INTERFACE)
target_link_libraries(fir-opt PRIVATE fir-opt-common)

target_link_libraries(fir-opt-common INTERFACE
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
)

mlir_target_link_libraries(fir-opt-common INTERFACE
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

if(FLANG_INCLUDE_TESTS)
  add_flang_executable(fir-test-opt fir-opt.cpp)
  llvm_update_compile_flags(fir-test-opt)
  target_compile_definitions(fir-test-opt PRIVATE FIR_TEST_OPT)
  target_link_libraries(fir-test-opt PRIVATE
    fir-opt-common
    FIRTestAnalysis
    FIRTestOpenACCInterfaces
    MLIRTestIR
  )
endif()
