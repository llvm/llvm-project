
add_library(FortranSemantics
  assignment.cpp
  attr.cpp
  canonicalize-do.cpp
  canonicalize-omp.cpp
  check-allocate.cpp
  check-arithmeticif.cpp
  check-call.cpp
  check-case.cpp
  check-coarray.cpp
  check-data.cpp
  check-deallocate.cpp
  check-declarations.cpp
  check-do-forall.cpp
  check-if-stmt.cpp
  check-io.cpp
  check-namelist.cpp
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

target_compile_features(FortranSemantics PUBLIC cxx_std_17)

target_link_libraries(FortranSemantics
  FortranCommon
  FortranEvaluate
  LLVMSupport
)

install (TARGETS FortranSemantics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
