#===-- lib/semantics/CMakeLists.txt ----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_library(FortranSemantics
  assignment.cpp
  attr.cpp
  canonicalize-do.cpp
  canonicalize-omp.cpp
  check-allocate.cpp
  check-arithmeticif.cpp
  check-call.cpp
  check-coarray.cpp
  check-data.cpp
  check-deallocate.cpp
  check-declarations.cpp
  check-do-forall.cpp
  check-if-stmt.cpp
  check-io.cpp
  check-nullify.cpp
  check-omp-structure.cpp
  check-purity.cpp
  check-return.cpp
  check-stop.cpp
  expression.cpp
  mod-file.cpp
  pointer-assignment.cpp
  program-tree.cpp
  resolve-labels.cpp
  resolve-names.cpp
  resolve-names-utils.cpp
  rewrite-parse-tree.cpp
  scope.cpp
  semantics.cpp
  symbol.cpp
  tools.cpp
  type.cpp
  unparse-with-symbols.cpp
)

target_link_libraries(FortranSemantics
  FortranCommon
  FortranEvaluate
)

install (TARGETS FortranSemantics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
