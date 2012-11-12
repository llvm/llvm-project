set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTooling
  ArgumentsAdjusters.cpp
  CommonOptionsParser.cpp
  CompilationDatabase.cpp
  FileMatchTrie.cpp
  JSONCompilationDatabase.cpp
  Refactoring.cpp
  RefactoringCallbacks.cpp
  Tooling.cpp
  )

add_dependencies(clangTooling
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangStmtNodes
  )

target_link_libraries(clangTooling
  clangBasic
  clangFrontend
  clangAST
  clangASTMatchers
  clangRewriteCore
  clangRewriteFrontend
  )
