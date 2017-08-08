set(LLVM_LINK_COMPONENTS
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
