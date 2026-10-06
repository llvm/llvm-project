include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

add_clang_tool(cir-translate
  cir-translate.cpp
)

clang_target_link_libraries(cir-translate
  PRIVATE
  clangCIR
  clangCIRLoweringDirectToLLVM
  CIRRegisterAllDialects
  MLIRCIR
  MLIRCIRTransforms
)

target_link_libraries(cir-translate
  PRIVATE
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRTranslateLib
  MLIRSupport
)
