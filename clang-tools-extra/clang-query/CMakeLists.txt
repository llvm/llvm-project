set(LLVM_LINK_COMPONENTS
  lineeditor
  support
  FrontendOpenMP
  )

add_clang_library(clangQuery STATIC
  Query.cpp
  QueryParser.cpp

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangQuery
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangSerialization
  )

add_subdirectory(tool)
