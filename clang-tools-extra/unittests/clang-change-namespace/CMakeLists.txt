set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CHANGE_NAMESPACE_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-change-namespace REALPATH)
include_directories(
  ${CHANGE_NAMESPACE_SOURCE_DIR}
  )

# We'd like clang/unittests/Tooling/RewriterTestContext.h in the test.
include_directories(${CLANG_SOURCE_DIR})

add_extra_unittest(ClangChangeNamespaceTests
  ChangeNamespaceTests.cpp
  )

clang_target_link_libraries(ClangChangeNamespaceTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  )
target_link_libraries(ClangChangeNamespaceTests
  PRIVATE
  clangChangeNamespace
  )
