# The interpreter can throw an exception from user input. The test binary needs
# to be compiled with exception support to expect and catch the thrown
# exception.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

add_clang_unittest(ClangReplInterpreterExceptionTests
  InterpreterExceptionTest.cpp
  )

llvm_update_compile_flags(ClangReplInterpreterExceptionTests)
target_link_libraries(ClangReplInterpreterExceptionTests PUBLIC
  clangAST
  clangBasic
  clangInterpreter
  clangFrontend
  )
add_dependencies(ClangReplInterpreterExceptionTests clang-resource-headers)
