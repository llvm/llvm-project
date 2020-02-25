#===-- lib/Parser/CMakeLists.txt -------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

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

target_link_libraries(FortranParser
  FortranCommon
)

install (TARGETS FortranParser
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
