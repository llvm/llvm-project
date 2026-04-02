add_subdirectory(Rewrite)

set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Option
  Plugins
  ProfileData
  Support
  TargetParser
  )

# config.h uses a #include guard that errors on double-inclusion, which
# breaks unity builds when multiple .cpp files include it.
set_source_files_properties(
  CompilerInstance.cpp
  CompilerInvocation.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangFrontend
  ASTConsumers.cpp
  ASTMerge.cpp
  ASTUnit.cpp
  ChainedDiagnosticConsumer.cpp
  ChainedIncludesSource.cpp
  CompilerInstance.cpp
  CompilerInvocation.cpp
  DependencyFile.cpp
  DependencyGraph.cpp
  DiagnosticRenderer.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  HeaderIncludeGen.cpp
  InitPreprocessor.cpp
  LayoutOverrideSource.cpp
  LogDiagnosticPrinter.cpp
  ModuleDependencyCollector.cpp
  MultiplexConsumer.cpp
  PrecompiledPreamble.cpp
  PrintPreprocessedOutput.cpp
  SARIFDiagnostic.cpp
  SARIFDiagnosticPrinter.cpp
  SerializedDiagnosticPrinter.cpp
  SerializedDiagnosticReader.cpp
  StandaloneDiagnostic.cpp
  TestModuleFileExtension.cpp
  TextDiagnostic.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp
  VerifyDiagnosticConsumer.cpp
  InterfaceStubFunctionsConsumer.cpp

  DEPENDS
  ClangDriverOptions
  intrinsics_gen

  LINK_LIBS
  clangAPINotes
  clangAST
  clangBasic
  clangOptions
  clangEdit
  clangLex
  clangParse
  clangSema
  clangSerialization
  )
