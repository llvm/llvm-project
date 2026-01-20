set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyGoogleModule STATIC
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
  UpgradeGoogletestCaseCheck.cpp
  UsingNamespaceDirectiveCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyMiscModule
  clangTidyModernizeModule
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
