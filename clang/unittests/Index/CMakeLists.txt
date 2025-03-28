set(LLVM_LINK_COMPONENTS
  )

add_clang_unittest(IndexTests
  IndexTests.cpp
  CLANG_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangIndex
  clangLex
  clangSerialization
  clangTooling
  LLVM_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  )

clang_target_link_libraries(IndexTests
  PRIVATE
  )
