set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANGD_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clangd REALPATH)
include_directories(
  ${CLANGD_SOURCE_DIR}
  )

add_extra_unittest(ClangdTests
  Annotations.cpp
  BackgroundIndexTests.cpp
  CancellationTests.cpp
  ClangdTests.cpp
  ClangdUnitTests.cpp
  CodeCompleteTests.cpp
  CodeCompletionStringsTests.cpp
  ContextTests.cpp
  DexTests.cpp
  DiagnosticsTests.cpp
  DraftStoreTests.cpp
  ExpectedTypeTest.cpp
  FileDistanceTests.cpp
  FileIndexTests.cpp
  FindSymbolsTests.cpp
  FSTests.cpp
  FunctionTests.cpp
  FuzzyMatchTests.cpp
  GlobalCompilationDatabaseTests.cpp
  HeadersTests.cpp
  IndexActionTests.cpp
  IndexTests.cpp
  JSONTransportTests.cpp
  QualityTests.cpp
  RIFFTests.cpp
  SelectionTests.cpp
  SerializationTests.cpp
  SourceCodeTests.cpp
  SymbolCollectorTests.cpp
  SymbolInfoTests.cpp
  SyncAPI.cpp
  TUSchedulerTests.cpp
  TestFS.cpp
  TestIndex.cpp
  TestTU.cpp
  ThreadingTests.cpp
  TraceTests.cpp
  TweakTests.cpp
  URITests.cpp
  XRefsTests.cpp

  $<TARGET_OBJECTS:obj.clangDaemonTweaks>
  )

target_link_libraries(ClangdTests
  PRIVATE
  clangAST
  clangBasic
  clangDaemon
  clangFormat
  clangFrontend
  clangIndex
  clangLex
  clangSema
  clangSerialization
  clangTidy
  clangTooling
  clangToolingCore
  clangToolingInclusions
  LLVMSupport
  LLVMTestingSupport
  )

if (CLANGD_BUILD_XPC)
  add_subdirectory(xpc)
endif ()
