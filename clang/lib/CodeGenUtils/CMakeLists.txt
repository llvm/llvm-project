set(LLVM_LINK_COMPONENTS
  CodeGen
  Core
  Demangle
  Support
  )

add_clang_library(clangCodeGenUtils
  BackendDiagnosticHandler.cpp
  CallUtils.cpp
  ClassUtils.cpp
  CodeGenUtils.cpp
  EHPersonality.cpp
  ExprUtils.cpp
  FunctionUtils.cpp
  ItaniumCXXABIUtils.cpp
  ModuleUtils.cpp
  RecordLayoutUtils.cpp
  StmtUtils.cpp
  TargetUtils.cpp

  LINK_LIBS
  clangAST
  clangBasic
  )
