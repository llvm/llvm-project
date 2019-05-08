set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(INCLUDE_FIXER_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-include-fixer REALPATH)
include_directories(
  ${INCLUDE_FIXER_SOURCE_DIR}
  )

# We'd like to clang/unittests/Tooling/RewriterTestContext.h in the test.
include_directories(${CLANG_SOURCE_DIR})

add_extra_unittest(ClangIncludeFixerTests
  IncludeFixerTest.cpp
  FuzzySymbolIndexTests.cpp
  )

target_link_libraries(ClangIncludeFixerTests
  PRIVATE
  clangBasic
  clangFormat
  clangFrontend
  clangIncludeFixer
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  findAllSymbols
  )

add_subdirectory(find-all-symbols)
