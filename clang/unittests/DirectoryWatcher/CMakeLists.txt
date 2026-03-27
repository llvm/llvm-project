if(APPLE OR CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME STREQUAL Windows)

  add_clang_unittest(DirectoryWatcherTests
    DirectoryWatcherTest.cpp
    CLANG_LIBS
    clangDirectoryWatcher
    LINK_LIBS
    LLVMTestingSupport
    LLVM_COMPONENTS
    Support
    )

endif()
