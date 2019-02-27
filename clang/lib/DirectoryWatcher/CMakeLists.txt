include(CheckIncludeFiles)

set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangDirectoryWatcher
  DirectoryWatcher.cpp
  )

if(APPLE)
  check_include_files("CoreServices/CoreServices.h" HAVE_CORESERVICES)
  if(HAVE_CORESERVICES)
    set(DIRECTORY_WATCHER_FLAGS "${DIRECTORY_WATCHER_FLAGS} -framework CoreServices")
    set_property(TARGET clangDirectoryWatcher APPEND_STRING PROPERTY
      LINK_FLAGS ${DIRECTORY_WATCHER_FLAGS})
  endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
  check_include_files("sys/inotify.h" HAVE_INOTIFY)
endif()

llvm_canonicalize_cmake_booleans(
  HAVE_CORESERVICES)
llvm_canonicalize_cmake_booleans(
  HAVE_INOTIFY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/Config.inc
)

if(BUILD_SHARED_LIBS AND APPLE AND HAVE_CORESERVICES)
  set(DIRECTORY_WATCHER_FLAGS "${DIRECTORY_WATCHER_FLAGS} -framework CoreServices")
  set_property(TARGET clangDirectoryWatcher APPEND_STRING PROPERTY
    LINK_FLAGS ${DIRECTORY_WATCHER_FLAGS})
endif()