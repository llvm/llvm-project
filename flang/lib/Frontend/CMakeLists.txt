get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

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

  MLIR_DEPS
  MLIRIR
  ${dialect_libs}
  ${extension_libs}

  MLIR_LIBS
  MLIRTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRImport
  ${dialect_libs}
  ${extension_libs}

  CLANG_LIBS
  clangBasic
  clangDriver
)

target_precompile_headers(flangFrontend PRIVATE
  [["flang/Parser/parsing.h"]]
  [["flang/Parser/parse-tree.h"]]
  [["flang/Parser/dump-parse-tree.h"]]
  [["flang/Lower/PFTBuilder.h"]]
  [["flang/Lower/Bridge.h"]]
)
