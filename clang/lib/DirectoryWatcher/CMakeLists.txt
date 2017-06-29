set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangDirectoryWatcher
  DirectoryWatcher.cpp

  LINK_LIBS
  clangBasic
  )
