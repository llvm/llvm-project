set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyGoogleModule STATIC
  AvoidCStyleCastsCheck.cpp
  AvoidNSObjectNewCheck.cpp
  AvoidThrowingObjCExceptionCheck.cpp
  AvoidUnderscoreInGoogletestNameCheck.cpp
  DefaultArgumentsCheck.cpp
  ExplicitConstructorCheck.cpp
  ExplicitMakePairCheck.cpp
  FloatTypesCheck.cpp
  FunctionNamingCheck.cpp
  GlobalNamesInHeadersCheck.cpp
  GlobalVariableDeclarationCheck.cpp
  GoogleTidyModule.cpp
  IntegerTypesCheck.cpp
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
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyGoogleModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
