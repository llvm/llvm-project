set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MC
  MCJIT
  Option
  Support
  native
  )

add_clang_executable(clang-interpreter
  main.cpp
  )

add_dependencies(clang-interpreter
  clang-headers
  )

target_link_libraries(clang-interpreter
  PRIVATE
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  )
