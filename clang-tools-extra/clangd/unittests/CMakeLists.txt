set(LLVM_LINK_COMPONENTS
  support
  AllTargetsInfos
  FrontendOpenMP
  )

if(CLANG_BUILT_STANDALONE)
  # LLVMTestingSupport library is needed for clangd tests.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      AND NOT TARGET LLVMTestingSupport)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      lib/Testing/Support)
  endif()
endif()

if (CLANGD_ENABLE_REMOTE)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/../index/remote)
  add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI=1)
  set(REMOTE_TEST_SOURCES remote/MarshallingTests.cpp)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../quality/CompletionModel.cmake)
gen_decision_forest(${CMAKE_CURRENT_SOURCE_DIR}/decision_forest_model DecisionForestRuntimeTest ::ns1::ns2::test::Example)

add_custom_target(ClangdUnitTests)
add_unittest(ClangdUnitTests ClangdTests
  Annotations.cpp
  ASTTests.cpp
  ASTSignalsTests.cpp
  BackgroundIndexTests.cpp
  CallHierarchyTests.cpp
  CanonicalIncludesTests.cpp
  ClangdTests.cpp
  ClangdLSPServerTests.cpp
  CodeCompleteTests.cpp
  CodeCompletionStringsTests.cpp
  CollectMacrosTests.cpp
  CompileCommandsTests.cpp
  CompilerTests.cpp
  ConfigCompileTests.cpp
  ConfigProviderTests.cpp
  ConfigYAMLTests.cpp
  DecisionForestTests.cpp
  DexTests.cpp
  DiagnosticsTests.cpp
  DraftStoreTests.cpp
  DumpASTTests.cpp
  ExpectedTypeTest.cpp
  FeatureModulesTests.cpp
  FileDistanceTests.cpp
  FileIndexTests.cpp
  FindSymbolsTests.cpp
  FindTargetTests.cpp
  FormatTests.cpp
  FSTests.cpp
  FuzzyMatchTests.cpp
  GlobalCompilationDatabaseTests.cpp
  HeadersTests.cpp
  HeaderSourceSwitchTests.cpp
  HoverTests.cpp
  IncludeCleanerTests.cpp
  IndexActionTests.cpp
  IndexTests.cpp
  InlayHintTests.cpp
  InsertionPointTests.cpp
  JSONTransportTests.cpp
  LoggerTests.cpp
  LSPBinderTests.cpp
  LSPClient.cpp
  ModulesTests.cpp
  ParsedASTTests.cpp
  PathMappingTests.cpp
  PreambleTests.cpp
  PrintASTTests.cpp
  ProjectAwareIndexTests.cpp
  QualityTests.cpp
  RenameTests.cpp
  RIFFTests.cpp
  SelectionTests.cpp
  SemanticHighlightingTests.cpp
  SemanticSelectionTests.cpp
  SerializationTests.cpp
  SourceCodeTests.cpp
  SymbolCollectorTests.cpp
  SymbolInfoTests.cpp
  SyncAPI.cpp
  TUSchedulerTests.cpp
  TestFS.cpp
  TestIndex.cpp
  TestTU.cpp
  TestWorkspace.cpp
  ThreadCrashReporterTests.cpp
  TidyProviderTests.cpp
  TypeHierarchyTests.cpp
  URITests.cpp
  XRefsTests.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/DecisionForestRuntimeTest.cpp

  support/CancellationTests.cpp
  support/ContextTests.cpp
  support/FunctionTests.cpp
  support/MarkupTests.cpp
  support/MemoryTreeTests.cpp
  support/PathTests.cpp
  support/ThreadingTests.cpp
  support/TestTracer.cpp
  support/TraceTests.cpp

  tweaks/AddUsingTests.cpp
  tweaks/AnnotateHighlightingsTests.cpp
  tweaks/DefineInlineTests.cpp
  tweaks/DefineOutlineTests.cpp
  tweaks/DumpASTTests.cpp
  tweaks/DumpRecordLayoutTests.cpp
  tweaks/DumpSymbolTests.cpp
  tweaks/ExpandAutoTypeTests.cpp
  tweaks/ExpandMacroTests.cpp
  tweaks/ExtractFunctionTests.cpp
  tweaks/ExtractVariableTests.cpp
  tweaks/ObjCLocalizeStringLiteralTests.cpp
  tweaks/ObjCMemberwiseInitializerTests.cpp
  tweaks/PopulateSwitchTests.cpp
  tweaks/RawStringLiteralTests.cpp
  tweaks/RemoveUsingNamespaceTests.cpp
  tweaks/ShowSelectionTreeTests.cpp
  tweaks/SwapIfBranchesTests.cpp
  tweaks/TweakTesting.cpp
  tweaks/TweakTests.cpp

  ${REMOTE_TEST_SOURCES}

  $<TARGET_OBJECTS:obj.clangDaemonTweaks>
  )

# Include generated ComletionModel headers.
target_include_directories(ClangdTests PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

clang_target_link_libraries(ClangdTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangIndex
  clangLex
  clangSema
  clangSerialization
  clangTooling
  clangToolingCore
  clangToolingInclusions
  clangToolingRefactoring
  clangToolingSyntax
  )
target_link_libraries(ClangdTests
  PRIVATE
  clangDaemon
  clangdSupport
  clangTidy
  LLVMTestingSupport
  )

if (CLANGD_ENABLE_REMOTE)
  target_link_libraries(ClangdTests
    PRIVATE
    clangdRemoteMarshalling
    clangdRemoteIndexProto)
endif()

if (CLANGD_BUILD_XPC)
  add_subdirectory(xpc)
endif ()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)
