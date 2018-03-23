set(LLVM_LINK_COMPONENTS support)

# Link Z3 if the user wants to build it.
if(CLANG_ANALYZER_WITH_Z3)
  set(Z3_LINK_FILES ${Z3_LIBRARIES})
else()
  set(Z3_LINK_FILES "")
endif()

add_clang_library(clangStaticAnalyzerCore
  APSIntType.cpp
  AnalysisManager.cpp
  AnalyzerOptions.cpp
  BasicValueFactory.cpp
  BlockCounter.cpp
  IssueHash.cpp
  BugReporter.cpp
  BugReporterVisitors.cpp
  CallEvent.cpp
  Checker.cpp
  CheckerContext.cpp
  CheckerHelpers.cpp
  CheckerManager.cpp
  CheckerRegistry.cpp
  CommonBugCategories.cpp
  ConstraintManager.cpp
  CoreEngine.cpp
  DynamicTypeMap.cpp
  Environment.cpp
  ExplodedGraph.cpp
  ExprEngine.cpp
  ExprEngineC.cpp
  ExprEngineCXX.cpp
  ExprEngineCallAndReturn.cpp
  ExprEngineObjC.cpp
  FunctionSummary.cpp
  HTMLDiagnostics.cpp
  LoopUnrolling.cpp
  LoopWidening.cpp
  MemRegion.cpp
  PathDiagnostic.cpp
  PlistDiagnostics.cpp
  ProgramState.cpp
  RangeConstraintManager.cpp
  RangedConstraintManager.cpp
  RegionStore.cpp
  SValBuilder.cpp
  SVals.cpp
  SimpleConstraintManager.cpp
  SimpleSValBuilder.cpp
  Store.cpp
  SubEngine.cpp
  SymbolManager.cpp
  WorkList.cpp
  Z3ConstraintManager.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangAnalysis
  clangBasic
  clangCrossTU
  clangLex
  clangRewrite
  ${Z3_LINK_FILES}
  )

if(CLANG_ANALYZER_WITH_Z3)
  target_include_directories(clangStaticAnalyzerCore SYSTEM
    PRIVATE
    ${Z3_INCLUDE_DIR}
    )
endif()
