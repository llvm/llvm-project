set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_unittest(FrontendTests
  FrontendActionTest.cpp
  )
target_link_libraries(FrontendTests
  clangFrontend
  )
