add_flang_library(flangTidyModernizeModule STATIC
  AvoidAssignStmt.cpp
  AvoidBackspaceStmt.cpp
  AvoidCommonBlocks.cpp
  AvoidDataConstructs.cpp
  AvoidPauseStmt.cpp
  ModernizeTidyModule.cpp

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
