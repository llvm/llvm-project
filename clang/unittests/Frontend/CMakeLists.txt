set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(FrontendTests
  FrontendActionTest.cpp
  CodeGenActionTest.cpp
  )
target_link_libraries(FrontendTests
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSema
  clangCodeGen
  )
