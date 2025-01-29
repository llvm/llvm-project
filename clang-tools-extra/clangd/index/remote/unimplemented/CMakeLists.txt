include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../)
# When compiled without Remote Index support, the real implementation index
# client is not present. Users will get a notification about this when trying
# to connect to remote index server instance.
add_clang_library(clangdRemoteIndex STATIC
  UnimplementedClient.cpp

  LINK_LIBS
  clangdSupport

  DEPENDS
  ClangDriverOptions
  )
