add_flang_library(flangTidyOpenMPModule STATIC
  AccumulatorRaceCheck.cpp
  OpenMPTidyModule.cpp

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
