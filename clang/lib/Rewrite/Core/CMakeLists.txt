add_clang_library(clangRewriteCore
  DeltaTree.cpp
  HTMLRewrite.cpp
  RewriteRope.cpp
  Rewriter.cpp
  TokenRewriter.cpp
  )

add_dependencies(clangRewriteCore
  ClangAttrClasses
  ClangAttrList
  ClangAttrParsedAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangStmtNodes
  )

target_link_libraries(clangRewriteCore
  clangBasic
  clangAST
  clangParse
  )
