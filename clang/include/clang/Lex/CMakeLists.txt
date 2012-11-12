clang_tablegen(AttrSpellings.inc -gen-clang-attr-spelling-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrSpellings
  DEPENDS AttrSpellings.inc)
