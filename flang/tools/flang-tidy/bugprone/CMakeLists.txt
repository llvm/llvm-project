add_flang_library(flangTidyBugproneModule STATIC
  ArithmeticGotoCheck.cpp
  BugproneTidyModule.cpp

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
