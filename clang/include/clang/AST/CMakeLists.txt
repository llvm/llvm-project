clang_tablegen(Attrs.inc -gen-clang-attr-classes
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrClasses)

clang_tablegen(AttrImpl.inc -gen-clang-attr-impl
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrImpl)

clang_tablegen(AttrTextNodeDump.inc -gen-clang-attr-text-node-dump
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrTextDump)

clang_tablegen(AttrNodeTraverse.inc -gen-clang-attr-node-traverse
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrTraverse)

clang_tablegen(AttrVisitor.inc -gen-clang-attr-ast-visitor
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrVisitor)

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

clang_tablegen(CommentHTMLNamedCharacterReferences.inc -gen-clang-comment-html-named-character-references
  SOURCE CommentHTMLNamedCharacterReferences.td
  TARGET ClangCommentHTMLNamedCharacterReferences)

clang_tablegen(CommentCommandInfo.inc -gen-clang-comment-command-info
  SOURCE CommentCommands.td
  TARGET ClangCommentCommandInfo)

clang_tablegen(CommentCommandList.inc -gen-clang-comment-command-list
  SOURCE CommentCommands.td
  TARGET ClangCommentCommandList)

clang_tablegen(StmtDataCollectors.inc -gen-clang-data-collectors
  SOURCE StmtDataCollectors.td
  TARGET StmtDataCollectors)
