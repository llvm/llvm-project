set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANGD_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clangd REALPATH)
get_filename_component(CLANGD_BINARY_DIR
  ${CMAKE_CURRENT_BINARY_DIR}/../../clangd REALPATH)
include_directories(
  ${CLANGD_SOURCE_DIR}
  ${CLANGD_BINARY_DIR}
  )

if(CLANG_BUILT_STANDALONE)
  # LLVMTestingSupport library is needed for clangd tests.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      AND NOT TARGET LLVMTestingSupport)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      lib/Testing/Support)
  endif()
endif()

add_custom_target(ClangdUnitTests)
add_unittest(ClangdUnitTests ClangdTests
  Annotations.cpp
  BackgroundIndexTests.cpp
  CancellationTests.cpp
  CanonicalIncludesTests.cpp
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
  FormattedStringTests.cpp
  FormatTests.cpp
  FSTests.cpp
  FunctionTests.cpp
  FuzzyMatchTests.cpp
  GlobalCompilationDatabaseTests.cpp
  HeadersTests.cpp
  IndexActionTests.cpp
  IndexTests.cpp
  JSONTransportTests.cpp
  PrintASTTests.cpp
  QualityTests.cpp
  RenameTests.cpp
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
  TypeHierarchyTests.cpp
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

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)
