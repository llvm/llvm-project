add_subdirectory(Dynamic)

set(LLVM_LINK_COMPONENTS
  Support
)

add_clang_library(clangASTMatchers
  ASTMatchFinder.cpp
  ASTMatchersInternal.cpp
  GtestMatchers.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  )

target_link_libraries(clangASTMatchers PUBLIC LLVMFrontendOpenMP)
