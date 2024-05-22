set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangTidyCERTModule
  CERTTidyModule.cpp
  CommandProcessorCheck.cpp
  DefaultOperatorNewAlignmentCheck.cpp
  DontModifyStdNamespaceCheck.cpp
  FloatLoopCounter.cpp
  LimitedRandomnessCheck.cpp
  MutatingCopyCheck.cpp
  NonTrivialTypesLibcMemoryCallsCheck.cpp
  PostfixOperatorCheck.cpp
  ProperlySeededRandomGeneratorCheck.cpp
  SetLongJmpCheck.cpp
  StaticObjectExceptionCheck.cpp
  StrToNumCheck.cpp
  ThrownExceptionTypeCheck.cpp
  VariadicFunctionDefCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyBugproneModule
  clangTidyConcurrencyModule
  clangTidyGoogleModule
  clangTidyMiscModule
  clangTidyPerformanceModule
  clangTidyReadabilityModule
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyCERTModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
