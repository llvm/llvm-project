set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyObjCModule
  AvoidNSErrorInitCheck.cpp
  ForbiddenSubclassingCheck.cpp
  MissingHashCheck.cpp
  ObjCTidyModule.cpp
  PropertyDeclarationCheck.cpp
  SuperSelfCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
