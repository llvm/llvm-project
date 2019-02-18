set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyPerformanceModule
  FasterStringFindCheck.cpp
  ForRangeCopyCheck.cpp
  ImplicitConversionInLoopCheck.cpp
  InefficientAlgorithmCheck.cpp
  InefficientStringConcatenationCheck.cpp
  InefficientVectorOperationCheck.cpp
  MoveConstArgCheck.cpp
  MoveConstructorInitCheck.cpp
  NoexceptMoveConstructorCheck.cpp
  PerformanceTidyModule.cpp
  TypePromotionInMathFnCheck.cpp
  UnnecessaryCopyInitialization.cpp
  UnnecessaryValueParamCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangAnalysis
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
