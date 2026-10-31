include_directories(${CLANG_CIR_MLIR_INCLUDE_DIRS})

if(CLANG_INCLUDE_TESTS)
  set(cir_opt_test_libs CIRTestAnalysis)
endif()

add_clang_tool(cir-opt
  cir-opt.cpp
)

if(CLANG_INCLUDE_TESTS)
  target_compile_definitions(cir-opt PRIVATE CLANG_INCLUDE_TESTS)
endif()

target_link_libraries(cir-opt
  PRIVATE
  ${cir_opt_test_libs}
  CIRRegisterAllDialects
  clangCIR
  clangCIRLoweringDirectToLLVM
  MLIRAnalysis
  MLIRCIR
  MLIRCIRTransforms
  MLIRDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROpenMPTransforms
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformUtils
)
