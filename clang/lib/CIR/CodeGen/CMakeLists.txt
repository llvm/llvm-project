set(
  LLVM_LINK_COMPONENTS
  Core
  Support
)

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRGenBuiltin.cpp
  CIRGenCXX.cpp
  CIRGenCXXABI.cpp
  CIRGenCall.cpp
  CIRGenClass.cpp
  CIRGenCleanup.cpp
  CIRGenCoroutine.cpp
  CIRGenDecl.cpp
  CIRGenDeclCXX.cpp
  CIRGenExpr.cpp
  CIRGenExprConst.cpp
  CIRGenExprAgg.cpp
  CIRGenExprCXX.cpp
  CIRGenExprScalar.cpp
  CIRGenFunction.cpp
  CIRGenItaniumCXXABI.cpp
  CIRGenModule.cpp
  CIRGenStmt.cpp
  CIRGenTBAA.cpp
  CIRGenTypes.cpp
  CIRGenerator.cpp
  CIRPasses.cpp
  CIRRecordLayoutBuilder.cpp
  TargetInfo.cpp

  DEPENDS
  MLIRCIR
  MLIRCIROpsIncGen
  ${dialect_libs}

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  ${dialect_libs}
  MLIRCIR
  MLIRCIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRSupport
  MLIRMemRefDialect
  MLIRTargetLLVMIRExport
  MLIRTransforms
)
