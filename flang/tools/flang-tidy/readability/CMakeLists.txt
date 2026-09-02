add_flang_library(flangTidyReadabilityModule STATIC
  FunctionCognitiveComplexityCheck.cpp
  FunctionSizeCheck.cpp
  ReadabilityTidyModule.cpp
  UnusedUSECheck.cpp

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
