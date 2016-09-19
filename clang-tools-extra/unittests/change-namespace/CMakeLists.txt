set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CHANGE_NAMESPACE_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../change-namespace REALPATH)
include_directories(
  ${CHANGE_NAMESPACE_SOURCE_DIR}
  )

# We'd like clang/unittests/Tooling/RewriterTestContext.h in the test.
include_directories(${CLANG_SOURCE_DIR})

add_extra_unittest(ChangeNamespaceTests
  ChangeNamespaceTests.cpp
  )

target_link_libraries(ChangeNamespaceTests
  clangAST
  clangASTMatchers
  clangBasic
  clangChangeNamespace
  clangFormat
  clangFrontend
  clangRewrite
  clangTooling
  clangToolingCore
  )
