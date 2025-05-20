add_flang_library(flangTidyOpenMPModule STATIC
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
