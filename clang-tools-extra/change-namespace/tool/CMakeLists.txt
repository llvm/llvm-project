include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(clang-change-namespace
  ClangChangeNamespace.cpp
  )
target_link_libraries(clang-change-namespace
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangChangeNamespace
  clangFormat
  clangFrontend
  clangRewrite
  clangTooling
  clangToolingCore
  )

install(TARGETS clang-change-namespace
  RUNTIME DESTINATION bin)
