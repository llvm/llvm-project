set(LLVM_LINK_COMPONENTS
  lineeditor
  support
  FrontendOpenMP
  )

add_clang_library(clangQuery
  Query.cpp
  QueryParser.cpp

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(clangQuery
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangTooling
  clangSerialization
  )

add_subdirectory(tool)
