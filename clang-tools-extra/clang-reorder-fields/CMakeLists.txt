set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  support
)

add_clang_library(clangReorderFields
  ReorderFieldsAction.cpp

  DEPENDS
  omp_gen
)

clang_target_link_libraries(clangReorderFields
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangIndex
  clangLex
  clangSerialization
  clangToolingCore
  )

add_subdirectory(tool)
