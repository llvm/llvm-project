set(LLVM_LINK_COMPONENTS support)

add_clang_executable(long-section-converter
  LongSectionConverter.cpp
  )
target_link_libraries(long-section-converter
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )
