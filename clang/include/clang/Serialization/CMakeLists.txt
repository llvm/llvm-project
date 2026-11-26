clang_tablegen(AttrPCHRead.inc -gen-clang-attr-pch-read
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrPCHRead)

clang_tablegen(AttrPCHWrite.inc -gen-clang-attr-pch-write
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrPCHWrite)
