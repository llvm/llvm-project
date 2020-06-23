set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyGoogleModule
  AvoidCStyleCastsCheck.cpp
  AvoidNSObjectNewCheck.cpp
  AvoidThrowingObjCExceptionCheck.cpp
  AvoidUnderscoreInGoogletestNameCheck.cpp
  DefaultArgumentsCheck.cpp
  ExplicitConstructorCheck.cpp
  ExplicitMakePairCheck.cpp
  FunctionNamingCheck.cpp
  GlobalNamesInHeadersCheck.cpp
  GlobalVariableDeclarationCheck.cpp
  GoogleTidyModule.cpp
  IntegerTypesCheck.cpp
  NonConstReferences.cpp
  OverloadedUnaryAndCheck.cpp
  TodoCommentCheck.cpp
  UnnamedNamespaceInHeaderCheck.cpp
  UpgradeGoogletestCaseCheck.cpp
  UsingNamespaceDirectiveCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(clangTidyGoogleModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
