set(LLVM_LINK_COMPONENTS Support)

include_directories("../lib") # FIXME: use public APIs instead.
add_clang_tool(clang-include-cleaner IncludeCleaner.cpp)
clang_target_link_libraries(clang-include-cleaner PRIVATE
  clangBasic
  clangLex
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )
target_link_libraries(clang-include-cleaner PRIVATE
  clangIncludeCleaner
  )

