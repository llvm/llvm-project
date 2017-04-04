set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANG_RENAME_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-rename REALPATH)
include_directories(
  ${CLANG_RENAME_SOURCE_DIR}
  )

# We'd like clang/unittests/Tooling/RewriterTestContext.h in the test.
include_directories(${CLANG_SOURCE_DIR})

add_extra_unittest(ClangRenameTests
  RenameClassTest.cpp
  )

target_link_libraries(ClangRenameTests
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangRename
  clangRewrite
  clangTooling
  clangToolingCore
  )
