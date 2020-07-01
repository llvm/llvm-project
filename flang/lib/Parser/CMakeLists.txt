
add_flang_library(FortranParser
  Fortran-parsers.cpp
  char-buffer.cpp
  char-block.cpp
  char-set.cpp
  characters.cpp
  debug-parser.cpp
  executable-parsers.cpp
  expr-parsers.cpp
  instrumented-parser.cpp
  io-parsers.cpp
  message.cpp
  openmp-parsers.cpp
  parse-tree.cpp
  parsing.cpp
  preprocessor.cpp
  prescan.cpp
  program-parsers.cpp
  provenance.cpp
  source.cpp
  token-sequence.cpp
  tools.cpp
  unparse.cpp
  user-state.cpp

  LINK_LIBS
  FortranCommon

  LINK_COMPONENTS
  Support

  DEPENDS
  omp_gen
)
