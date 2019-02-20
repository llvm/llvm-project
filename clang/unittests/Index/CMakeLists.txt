set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  )

add_clang_unittest(IndexTests
  IndexTests.cpp
  )

target_link_libraries(IndexTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangIndex
  clangLex
  clangSerialization
  clangTooling
  )
