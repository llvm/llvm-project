set(LLVM_LINK_COMPONENTS
  lineeditor
  support
  )

add_clang_library(clangQuery
  Query.cpp
  QueryParser.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangSerialization
  )

add_subdirectory(tool)
