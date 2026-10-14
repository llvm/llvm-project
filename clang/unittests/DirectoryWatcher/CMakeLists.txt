include(CheckIncludeFiles)

if(APPLE)
  check_include_files("CoreServices/CoreServices.h" HAVE_CORESERVICES)
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  check_include_files("sys/inotify.h" HAVE_INOTIFY)
endif()

if((APPLE AND HAVE_CORESERVICES) OR
   (CMAKE_SYSTEM_NAME MATCHES "Linux" AND HAVE_INOTIFY) OR
   CMAKE_SYSTEM_NAME MATCHES "Windows")

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
