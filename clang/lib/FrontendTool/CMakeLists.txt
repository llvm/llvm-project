add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )

add_dependencies(clangFrontendTool
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangDriverOptions
  )

target_link_libraries(clangFrontendTool
  clangDriver
  clangFrontend
  clangRewriteCore
  clangRewriteFrontend
  clangCodeGen
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  clangARCMigrate
  )
