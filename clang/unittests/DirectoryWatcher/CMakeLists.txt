set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(DirectoryWatcherTests
  DirectoryWatcherTest.cpp
  )

target_link_libraries(DirectoryWatcherTests
  PRIVATE
  clangDirectoryWatcher
  clangBasic
  )
