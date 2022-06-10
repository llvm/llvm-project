set(LLVM_LINK_COMPONENTS
  Support
  TestingSupport
  )

add_custom_target(ClangIncludeCleanerUnitTests)
add_unittest(ClangIncludeCleanerUnitTests ClangIncludeCleanerTests
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
  )

target_link_libraries(ClangIncludeCleanerTests
  PRIVATE
  clangIncludeCleaner
  clangTesting
  LLVMTestingSupport
  )

