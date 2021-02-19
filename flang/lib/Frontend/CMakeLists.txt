add_flang_library(flangFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  TextDiagnosticPrinter.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnostic.cpp

  DEPENDS
  clangBasic

  LINK_LIBS
  FortranParser
  FortranSemantics
  FortranEvaluate
  FortranCommon
  clangBasic
  clangDriver

  LINK_COMPONENTS
  Option
  Support
  FrontendOpenACC
  FrontendOpenMP
)
