set(LLVM_LINK_COMPONENTS support)

add_clang_executable(feature-extractor
  main.cpp
)

target_link_libraries(feature-extractor
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
)
