set(
  LLVM_LINK_COMPONENTS
  Core
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRGenerator.cpp
  CIRGenAsm.cpp
  CIRGenAtomic.cpp
  CIRGenBuilder.cpp
  CIRGenBuiltin.cpp
  CIRGenBuiltinAArch64.cpp
  CIRGenBuiltinAMDGPU.cpp
  CIRGenAMDGPU.cpp
  CIRGenBuiltinRISCV.cpp
  CIRGenBuiltinX86.cpp
  CIRGenCall.cpp
  CIRGenClass.cpp
  CIRGenCleanup.cpp
  CIRGenCoroutine.cpp
  CIRGenCUDANV.cpp
  CIRGenCUDARuntime.cpp
  CIRGenCXX.cpp
  CIRGenCXXABI.cpp
  CIRGenDecl.cpp
  CIRGenDeclCXX.cpp
  CIRGenDeclOpenACC.cpp
  CIRGenDeclOpenMP.cpp
  CIRGenException.cpp
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
  CIRGenOpenACCRecipe.cpp
  CIRGenOpenMPClause.cpp
  CIRGenPointerAuth.cpp
  CIRGenRecordLayoutBuilder.cpp
  CIRGenStmt.cpp
  CIRGenStmtOpenACC.cpp
  CIRGenStmtOpenACCLoop.cpp
  CIRGenStmtOpenMP.cpp
  CIRGenTypes.cpp
  CIRGenVTables.cpp
  TargetInfo.cpp
  Targets/AMDGPU.cpp

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
  CIROpenMPSupport
  MLIRCIR
  MLIRCIRInterfaces
  MLIRTargetLLVMIRImport
)


# CIRGenExprComplex.cpp references both cir::ComplexType/CastKind and
# clang::ComplexType/CastKind which become ambiguous in unity builds.
set_source_files_properties(CIRGenExprComplex.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# CIRGenOpenACCRecipe.h has no include guards and defines classes; including it
# from both CIRGenOpenACCClause.cpp and CIRGenOpenACCRecipe.cpp in the same
# unity TU causes redefinition errors.
set_source_files_properties(
  CIRGenOpenACCClause.cpp
  CIRGenOpenACCRecipe.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# CIRGenTypes.cpp references cir::Type/RecordType which conflict with
# clang::Type/RecordType from other files in the unity TU.
set_source_files_properties(CIRGenTypes.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
