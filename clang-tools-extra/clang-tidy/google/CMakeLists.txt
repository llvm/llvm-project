set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyGoogleModule
  AvoidCStyleCastsCheck.cpp
  DefaultArgumentsCheck.cpp
  ExplicitConstructorCheck.cpp
  ExplicitMakePairCheck.cpp
  GlobalNamesInHeadersCheck.cpp
  GlobalVariableDeclarationCheck.cpp
  GoogleTidyModule.cpp
  IntegerTypesCheck.cpp
  NonConstReferences.cpp
  OverloadedUnaryAndCheck.cpp
  StringReferenceMemberCheck.cpp
  TodoCommentCheck.cpp
  UnnamedNamespaceInHeaderCheck.cpp
  UsingNamespaceDirectiveCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils
  )
