add_subdirectory(Rewrite)

set(LLVM_LINK_COMPONENTS
  BitReader
  Option
  ProfileData
  Support
  )

set(optional_deps intrinsics_gen)
if (CLANG_BUILT_STANDALONE)
  set(optional_deps)
endif()

add_clang_library(clangFrontend
  ASTConsumers.cpp
  ASTMerge.cpp
  ASTUnit.cpp
  CacheTokens.cpp
  ChainedDiagnosticConsumer.cpp
  ChainedIncludesSource.cpp
  CodeGenOptions.cpp
  CompilerInstance.cpp
  CompilerInvocation.cpp
  CreateInvocationFromCommandLine.cpp
  DependencyFile.cpp
  DependencyGraph.cpp
  DiagnosticRenderer.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  FrontendTiming.cpp
  HeaderIncludeGen.cpp
  InitHeaderSearch.cpp
  InitPreprocessor.cpp
  LangStandards.cpp
  LayoutOverrideSource.cpp
  LogDiagnosticPrinter.cpp
  ModuleDependencyCollector.cpp
  MultiplexConsumer.cpp
  PCHContainerOperations.cpp
  PrecompiledPreamble.cpp
  PrintPreprocessedOutput.cpp
  SerializedDiagnosticPrinter.cpp
  SerializedDiagnosticReader.cpp
  TestModuleFileExtension.cpp
  TextDiagnostic.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp
  VerifyDiagnosticConsumer.cpp

  DEPENDS
  ClangDriverOptions
  ${optional_deps}

  LINK_LIBS
  clangAST
  clangBasic
  clangDriver
  clangEdit
  clangLex
  clangParse
  clangSema
  clangSerialization
  )
