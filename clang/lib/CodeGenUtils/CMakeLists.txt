set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangCodeGenUtils
  CodeGenUtils.cpp

  LINK_LIBS
  clangAST
  clangBasic
  )
