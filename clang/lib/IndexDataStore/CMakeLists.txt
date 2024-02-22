set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangIndexDataStore
  IndexDataStore.cpp

  LINK_LIBS
  clangDirectoryWatcher
  clangIndex
  )
