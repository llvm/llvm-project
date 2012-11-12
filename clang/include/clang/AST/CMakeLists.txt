clang_tablegen(Attrs.inc -gen-clang-attr-classes
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrClasses)

clang_tablegen(AttrImpl.inc -gen-clang-attr-impl
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrImpl)

clang_tablegen(StmtNodes.inc -gen-clang-stmt-nodes
  SOURCE ../Basic/StmtNodes.td
  TARGET ClangStmtNodes)

clang_tablegen(DeclNodes.inc -gen-clang-decl-nodes
  SOURCE ../Basic/DeclNodes.td
  TARGET ClangDeclNodes)

clang_tablegen(CommentNodes.inc -gen-clang-comment-nodes
  SOURCE ../Basic/CommentNodes.td
  TARGET ClangCommentNodes)

clang_tablegen(CommentHTMLTags.inc -gen-clang-comment-html-tags
  SOURCE CommentHTMLTags.td
  TARGET ClangCommentHTMLTags)

clang_tablegen(CommentHTMLTagsProperties.inc -gen-clang-comment-html-tags-properties
  SOURCE CommentHTMLTags.td
  TARGET ClangCommentHTMLTagsProperties)

clang_tablegen(CommentCommandInfo.inc -gen-clang-comment-command-info
  SOURCE CommentCommands.td
  TARGET ClangCommentCommandInfo)

