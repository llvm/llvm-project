set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyPerformanceModule
  FasterStringFindCheck.cpp
  ForRangeCopyCheck.cpp
  ImplicitConversionInLoopCheck.cpp
  InefficientAlgorithmCheck.cpp
  InefficientStringConcatenationCheck.cpp
  InefficientVectorOperationCheck.cpp
  MoveConstArgCheck.cpp
  MoveConstructorInitCheck.cpp
  NoAutomaticMoveCheck.cpp
  NoexceptMoveConstructorCheck.cpp
  PerformanceTidyModule.cpp
  TriviallyDestructibleCheck.cpp
  TypePromotionInMathFnCheck.cpp
  UnnecessaryCopyInitialization.cpp
  UnnecessaryValueParamCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils
  )

clang_target_link_libraries(clangTidyPerformanceModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangAnalysis
  clangBasic
  clangLex
  )
