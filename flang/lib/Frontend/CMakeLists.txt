get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_flang_library(flangFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  CodeGenOptions.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  LangOptions.cpp
  TextDiagnosticPrinter.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnostic.cpp

  DEPENDS
  CUFDialect
  FIRDialect
  FIROptCodeGenPassIncGen
  FIROptTransformsPassIncGen
  HLFIRDialect
  MLIRIR
  ${dialect_libs}
  ${extension_libs}

  LINK_LIBS
  CUFDialect
  FortranParser
  FortranSemantics
  FortranEvaluate
  FortranCommon
  FortranLower
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FIRBuilder
  FIRCodeGen
  FIRTransforms
  HLFIRDialect
  HLFIRTransforms
  MLIRTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRImport
  ${dialect_libs}
  ${extension_libs}

  LINK_COMPONENTS
  Passes
  Analysis
  Extensions
  IRPrinter
  IRReader
  Option
  Support
  Target
  TargetParser
  FrontendDriver
  FrontendOpenACC
  FrontendOpenMP

  CLANG_LIBS
  clangBasic
  clangDriver
)
