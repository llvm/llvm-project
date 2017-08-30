set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  Option
  )

add_clang_unittest(ClangDriverTests
  DistroTest.cpp
  ToolChainTest.cpp
  MultilibTest.cpp
  )

target_link_libraries(ClangDriverTests
  clangDriver
  clangBasic
  )
