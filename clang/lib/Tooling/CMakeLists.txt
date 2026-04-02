set(LLVM_LINK_COMPONENTS
  Option
  FrontendOpenMP
  Support
  TargetParser
  )

add_subdirectory(Core)
add_subdirectory(Inclusions)
add_subdirectory(Refactoring)
add_subdirectory(ASTDiff)
add_subdirectory(Syntax)
add_subdirectory(Transformer)

# CompilationDatabase.cpp and DependencyScanningTool.cpp reference
# clang::driver which becomes ambiguous when combined with other files
# in a unity build.
set_source_files_properties(
  CompilationDatabase.cpp
  DependencyScanningTool.cpp
  Tooling.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangTooling
  AllTUsExecution.cpp
  ArgumentsAdjusters.cpp
  CommonOptionsParser.cpp
  CompilationDatabase.cpp
  DependencyScanningTool.cpp
  Execution.cpp
  ExpandResponseFilesCompilationDatabase.cpp
  FileMatchTrie.cpp
  FixIt.cpp
  GuessTargetAndModeCompilationDatabase.cpp
  InterpolatingCompilationDatabase.cpp
  JSONCompilationDatabase.cpp
  LocateToolCompilationDatabase.cpp
  Refactoring.cpp
  RefactoringCallbacks.cpp
  StandaloneExecution.cpp
  Tooling.cpp

  DEPENDS
  ClangDriverOptions
  omp_gen

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangDependencyScanning
  clangDriver
  clangOptions
  clangFormat
  clangFrontend
  clangLex
  clangRewrite
  clangSerialization
  clangToolingCore
  ${LLVM_PTHREAD_LIB}
  )
