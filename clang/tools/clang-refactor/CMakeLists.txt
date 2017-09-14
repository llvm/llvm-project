set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_executable(clang-refactor
  ClangRefactor.cpp
  TestSupport.cpp
  )

target_link_libraries(clang-refactor
  clangAST
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangRewrite
  clangTooling
  clangToolingCore
  clangToolingRefactor
  )

install(TARGETS clang-refactor RUNTIME DESTINATION bin)
