include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

if(CLANG_INCLUDE_TESTS)
  set(cir_opt_test_libs CIRTestAnalysis)
endif()

add_clang_tool(cir-opt
  cir-opt.cpp
)

if(CLANG_INCLUDE_TESTS)
  target_compile_definitions(cir-opt PRIVATE CLANG_INCLUDE_TESTS)
endif()

clang_target_link_libraries(cir-opt
  PRIVATE
  clangCIR
  clangCIRLoweringDirectToLLVM
  CIRRegisterAllDialects
  MLIRCIR
  MLIRCIRTransforms
  ${cir_opt_test_libs}
)

target_link_libraries(cir-opt
  PRIVATE
  MLIRAnalysis
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
