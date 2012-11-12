clang_tablegen(AttrTemplateInstantiate.inc -gen-clang-attr-template-instantiate
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrTemplateInstantiate)
  
clang_tablegen(AttrParsedAttrList.inc -gen-clang-attr-parsed-attr-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrParsedAttrList)

clang_tablegen(AttrParsedAttrKinds.inc -gen-clang-attr-parsed-attr-kinds
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrParsedAttrKinds)