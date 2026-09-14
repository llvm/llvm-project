set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangCodeGenUtils
  CallUtils.cpp
  ClassUtils.cpp
  EHPersonality.cpp
  ExprUtils.cpp
  FunctionUtils.cpp
  ItaniumCXXABIUtils.cpp
  ModuleUtils.cpp
  RecordLayoutUtils.cpp
  TargetUtils.cpp

  LINK_LIBS
  clangAST
  clangBasic
  )
