if(APPLE OR CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME STREQUAL Windows)

  set(LLVM_LINK_COMPONENTS
    Support
    )

  add_clang_unittest(DirectoryWatcherTests
    DirectoryWatcherTest.cpp
    )

  target_link_libraries(DirectoryWatcherTests
    PRIVATE
    LLVMTestingSupport
    clangDirectoryWatcher
    clangBasic
    )

endif()
