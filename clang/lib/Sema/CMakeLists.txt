set(LLVM_LINK_COMPONENTS
  Core
  FrontendOpenMP
  Support
  )

clang_tablegen(OpenCLBuiltins.inc -gen-clang-opencl-builtins
  SOURCE OpenCLBuiltins.td
  TARGET ClangOpenCLBuiltinsImpl
  )

add_clang_library(clangSema
  AnalysisBasedWarnings.cpp
  CodeCompleteConsumer.cpp
  DeclSpec.cpp
  DelayedDiagnostic.cpp
  IdentifierResolver.cpp
  JumpDiagnostics.cpp
  MultiplexExternalSemaSource.cpp
  ParsedAttr.cpp
  Scope.cpp
  ScopeInfo.cpp
  Sema.cpp
  SemaAccess.cpp
  SemaAttr.cpp
  SemaAvailability.cpp
  SemaCXXScopeSpec.cpp
  SemaCast.cpp
  SemaChecking.cpp
  SemaCodeComplete.cpp
  SemaConcept.cpp
  SemaConsumer.cpp
  SemaCoroutine.cpp
  SemaCUDA.cpp
  SemaDecl.cpp
  SemaDeclAttr.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExceptionSpec.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprMember.cpp
  SemaExprObjC.cpp
  SemaFixItUtils.cpp
  SemaInit.cpp
  SemaLambda.cpp
  SemaLookup.cpp
  SemaModule.cpp
  SemaObjCProperty.cpp
  SemaOpenMP.cpp
  SemaOverload.cpp
  SemaPseudoObject.cpp
  SemaStmt.cpp
  SemaStmtAsm.cpp
  SemaStmtAttr.cpp
  SemaSYCL.cpp
  SemaTemplate.cpp
  SemaTemplateDeduction.cpp
  SemaTemplateInstantiate.cpp
  SemaTemplateInstantiateDecl.cpp
  SemaTemplateVariadic.cpp
  SemaType.cpp
  TypeLocBuilder.cpp

  DEPENDS
  ClangOpenCLBuiltinsImpl
  omp_gen

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangEdit
  clangLex
  )
