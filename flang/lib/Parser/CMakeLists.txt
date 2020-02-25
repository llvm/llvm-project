
add_library(FortranParser
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
)

target_compile_features(FortranParser PRIVATE cxx_std_17)

target_link_libraries(FortranParser
  FortranCommon
  LLVMSupport
)

install (TARGETS FortranParser
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
