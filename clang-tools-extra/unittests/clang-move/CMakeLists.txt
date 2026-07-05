set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

get_filename_component(INCLUDE_FIXER_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-move REALPATH)
include_directories(
  ${INCLUDE_FIXER_SOURCE_DIR}
  )

# We'd like to clang/unittests/Tooling/RewriterTestContext.h in the test.
include_directories(${CLANG_SOURCE_DIR})

add_extra_unittest(ClangMoveTests
  ClangMoveTests.cpp
  )

clang_target_link_libraries(ClangMoveTests
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
target_link_libraries(ClangMoveTests
  PRIVATE
  clangMove
  )
