set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(DirectoryWatcherTests
  DirectoryWatcherTest.cpp
  )

target_link_libraries(DirectoryWatcherTests
  PRIVATE
  clangDirectoryWatcher
  )

if(APPLE)
  check_include_files("CoreServices/CoreServices.h" HAVE_CORESERVICES_H)
  if(HAVE_CORESERVICES_H)
    set(DWT_LINK_FLAGS "${DWT_LINK_FLAGS} -framework CoreServices")
    set_property(TARGET DirectoryWatcherTests APPEND_STRING PROPERTY
                 LINK_FLAGS ${DWT_LINK_FLAGS})
  endif()
endif()
