add_clang_library(clangEdit
  Commit.cpp
  EditedSource.cpp
  RewriteObjCFoundationAPI.cpp
  )

add_dependencies(clangEdit
  ClangAttrClasses
  ClangAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangStmtNodes
  )

target_link_libraries(clangEdit
  clangBasic
  clangAST
  clangLex
  )
