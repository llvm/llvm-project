add_clang_library(clangRewriteFrontend
  FixItRewriter.cpp
  FrontendActions.cpp
  HTMLPrint.cpp
  InclusionRewriter.cpp
  RewriteMacros.cpp
  RewriteModernObjC.cpp
  RewriteObjC.cpp
  RewriteTest.cpp
  )

add_dependencies(clangRewriteFrontend
  ClangAttrClasses
  ClangAttrList
  ClangAttrParsedAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangStmtNodes
  )

target_link_libraries(clangRewriteFrontend
  clangBasic
  clangAST
  clangParse
  clangFrontend
  )
