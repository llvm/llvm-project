get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

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
  FIRBuilder
  FIRDialect
  FIRSupport
  FIROptTransformsPassIncGen
  MLIRIR
  ${dialect_libs}

  LINK_LIBS
  FortranParser
  FortranSemantics
  FortranEvaluate
  FortranCommon
  FortranLower
  clangBasic
  clangDriver
  FIRDialect
  FIRSupport
  FIRBuilder
  FIRCodeGen
  FIRTransforms
  MLIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToControlFlow
  ${dialect_libs}

  LINK_COMPONENTS
  Passes
  Analysis
  Option
  Support
  Target
  FrontendOpenACC
  FrontendOpenMP
)
