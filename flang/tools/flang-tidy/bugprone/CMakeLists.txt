add_flang_library(flangTidyBugproneModule STATIC
  ArithmeticGotoCheck.cpp
  ArithmeticIfStmtCheck.cpp
  BugproneTidyModule.cpp
  ContiguousArrayCheck.cpp
  ImplicitDeclCheck.cpp
  ImpliedSaveCheck.cpp
  MismatchedIntentCheck.cpp
  MissingActionCheck.cpp
  MissingDefaultCheck.cpp
  PrecisionLossCheck.cpp
  ShortCircuitCheck.cpp
  SubprogramTrampolineCheck.cpp
  UndeclaredProcCheck.cpp
  UnusedIntentCheck.cpp

  LINK_LIBS
  FortranSupport
  FortranParser
  FortranEvaluate
  FortranSemantics
  LLVMSupport
  flangTidy

  LINK_COMPONENTS
  Support
  FrontendOpenMP
  FrontendOpenACC
  TargetParser
  )
