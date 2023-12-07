set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangChangeNamespace
  ChangeNamespace.cpp

  DEPENDS
  omp_gen
  ClangDriverOptions
)

clang_target_link_libraries(clangChangeNamespace
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  clangToolingCore
  )

add_subdirectory(tool)
