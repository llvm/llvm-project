# The interpreter can throw an exception from user input. The test binary needs
# to be compiled with exception support to catch the thrown exception.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

add_distinct_clang_unittest(ClangReplInterpreterExceptionTests
  InterpreterExceptionTest.cpp
  EXPORT_SYMBOLS

  CLANG_LIBS
  clangAST
  clangBasic
  clangInterpreter
  clangFrontend

  LLVM_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  OrcJIT
  Support
  )

llvm_update_compile_flags(ClangReplInterpreterExceptionTests)
add_dependencies(ClangReplInterpreterExceptionTests clang-resource-headers)
