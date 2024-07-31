set(LLVM_LINK_COMPONENTS
  Support
  TestingSupport
  )

add_custom_target(ClangIncludeCleanerUnitTests)
set_target_properties(ClangIncludeCleanerUnitTests PROPERTIES FOLDER "Clang Tools Extra/Tests")
add_unittest(ClangIncludeCleanerUnitTests ClangIncludeCleanerTests
  AnalysisTest.cpp
  FindHeadersTest.cpp
  IncludeSpellerTest.cpp
  LocateSymbolTest.cpp
  RecordTest.cpp
  TypesTest.cpp
  WalkASTTest.cpp
)

target_include_directories(ClangIncludeCleanerTests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

clang_target_link_libraries(ClangIncludeCleanerTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangFormat
  clangLex
  clangSerialization
  clangToolingInclusionsStdlib
  )

target_link_libraries(ClangIncludeCleanerTests
  PRIVATE
  clangIncludeCleaner
  clangTesting
  LLVMTestingAnnotations
  LLVMTestingSupport
  )

