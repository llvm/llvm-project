set(LLVM_LINK_COMPONENTS
  support
  )

# We'd like clang/unittests/Tooling/RewriterTestContext.h in the test.
include_directories(${CLANG_SOURCE_DIR})

add_clang_unittest(ClangRenameTests
  RenameClassTest.cpp
  RenameEnumTest.cpp
  RenameAliasTest.cpp
  RenameMemberTest.cpp
  RenameFunctionTest.cpp
  )

target_link_libraries(ClangRenameTests
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangTooling
  clangToolingCore
  clangToolingRefactor
  )
