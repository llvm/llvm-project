clang_tablegen(Checkers.inc -gen-clang-sa-checkers
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../../
  SOURCE Checkers.td
  TARGET ClangSACheckers)
