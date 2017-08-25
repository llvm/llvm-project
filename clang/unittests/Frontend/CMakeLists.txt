set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(FrontendTests
  ASTUnitTest.cpp
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
