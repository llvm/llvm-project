set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyObjCModule
  AvoidNSErrorInitCheck.cpp
  AvoidSpinlockCheck.cpp
  ForbiddenSubclassingCheck.cpp
  ObjCTidyModule.cpp
  PropertyDeclarationCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
