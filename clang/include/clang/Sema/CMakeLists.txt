clang_tablegen(AttrTemplateInstantiate.inc -gen-clang-attr-template-instantiate
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrTemplateInstantiate)

clang_tablegen(AttrParsedAttrKinds.inc -gen-clang-attr-parsed-attr-kinds
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrParsedAttrKinds)

clang_tablegen(AttrIsTypeDependent.inc -gen-clang-attr-is-type-dependent
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrIsTypeDependent)

clang_tablegen(AttrSpellingListIndex.inc -gen-clang-attr-spelling-index
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrSpellingListIndex)

clang_tablegen(AttrParsedAttrImpl.inc -gen-clang-attr-parsed-attr-impl
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrParsedAttrImpl)
