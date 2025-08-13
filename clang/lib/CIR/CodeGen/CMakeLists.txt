set(
  LLVM_LINK_COMPONENTS
  Core
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRGenerator.cpp
  CIRGenAtomic.cpp
  CIRGenBuilder.cpp
  CIRGenCall.cpp
  CIRGenClass.cpp
  CIRGenCleanup.cpp
  CIRGenCXX.cpp
  CIRGenCXXABI.cpp
  CIRGenCXXExpr.cpp
  CIRGenBuiltin.cpp
  CIRGenDecl.cpp
  CIRGenDeclOpenACC.cpp
  CIRGenExpr.cpp
  CIRGenExprAggregate.cpp
  CIRGenExprComplex.cpp
  CIRGenExprConstant.cpp
  CIRGenExprCXX.cpp
  CIRGenExprScalar.cpp
  CIRGenFunction.cpp
  CIRGenItaniumCXXABI.cpp
  CIRGenModule.cpp
  CIRGenOpenACC.cpp
  CIRGenOpenACCClause.cpp
  CIRGenRecordLayoutBuilder.cpp
  CIRGenStmt.cpp
  CIRGenStmtOpenACC.cpp
  CIRGenStmtOpenACCLoop.cpp
  CIRGenTypes.cpp
  CIRGenVTables.cpp
  TargetInfo.cpp

  DEPENDS
  MLIRCIR
  MLIRCIROpInterfacesIncGen
  ${dialect_libs}

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  ${dialect_libs}
  CIROpenACCSupport
  MLIRCIR
  MLIRCIRInterfaces
  MLIRTargetLLVMIRImport
)
