set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  Option
  )

add_clang_unittest(ClangDriverTests
  DistroTest.cpp
  ToolChainTest.cpp
  ModuleCacheTest.cpp
  MultilibTest.cpp
  )

clang_target_link_libraries(ClangDriverTests
  PRIVATE
  clangDriver
  clangBasic
  )
