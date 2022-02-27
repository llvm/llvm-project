set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRGenCall.cpp
  CIRGenerator.cpp
  CIRGenCXXABI.cpp
  CIRGenExprScalar.cpp
  CIRGenFunction.cpp
  CIRGenModule.cpp
  CIRGenTypes.cpp
  CIRRecordLayoutBuilder.cpp
  LowerToLLVM.cpp

  DEPENDS
  MLIRCIROpsIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  ${dialect_libs}
  MLIRCIR
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
