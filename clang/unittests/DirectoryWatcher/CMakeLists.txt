if(APPLE OR CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME STREQUAL Windows)

  add_clang_unittest(DirectoryWatcherTests
    DirectoryWatcherTest.cpp
    LINK_LIBS
    LLVMTestingSupport
    clangDirectoryWatcher
    LLVM_COMPONENTS
    Support
    )

endif()
