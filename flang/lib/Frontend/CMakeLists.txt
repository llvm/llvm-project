get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY AIIR_EXTENSION_LIBS)

add_flang_library(flangFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  CodeGenOptions.cpp
  ParserActions.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  TextDiagnosticPrinter.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnostic.cpp

  DISABLE_PCH_REUSE
  PRECOMPILE_HEADERS
  [["flang/Parser/parsing.h"]]
  [["flang/Parser/parse-tree.h"]]
  [["flang/Parser/dump-parse-tree.h"]]
  [["flang/Lower/PFTBuilder.h"]]
  [["flang/Lower/Bridge.h"]]

  DEPENDS
  CUFDialect
  FIRDialect
  FIROptCodeGenPassIncGen
  FIROptTransformsPassIncGen
  HLFIRDialect
  MIFDialect

  LINK_LIBS
  CUFDialect
  FortranParser
  FortranSemantics
  FortranEvaluate
  FortranSupport
  FortranLower
  FortranSupport
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FIRBuilder
  FIRCodeGen
  FIRCodeGenDialect
  FIRTransforms
  HLFIRDialect
  HLFIRTransforms
  flangPasses
  FIROpenACCSupport
  FIROpenMPSupport
  FlangOpenMPTransforms
  MIFDialect

  LINK_COMPONENTS
  Passes
  Plugins
  Analysis
  BitReader
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

  AIIR_DEPS
  AIIRIR
  ${dialect_libs}
  ${extension_libs}

  AIIR_LIBS
  AIIRTransforms
  AIIRBuiltinToLLVMIRTranslation
  AIIRLLVMToLLVMIRTranslation
  AIIRSCFToControlFlow
  AIIRTargetLLVMIRImport
  ${dialect_libs}
  ${extension_libs}

  CLANG_LIBS
  clangBasic
  clangOptions
)
