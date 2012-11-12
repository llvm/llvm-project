clang_tablegen(AttrLateParsed.inc -gen-clang-attr-late-parsed-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrLateParsed)
