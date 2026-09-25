set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangCodeGenUtils
  CallUtils.cpp
  ClassUtils.cpp
  CodeGenUtils.cpp
  ExprUtils.cpp
  FunctionUtils.cpp
  ItaniumCXXABIUtils.cpp

  LINK_LIBS
  clangAST
  clangBasic
  )
