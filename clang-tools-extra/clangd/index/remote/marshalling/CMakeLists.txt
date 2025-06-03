add_clang_library(clangdRemoteMarshalling STATIC
  Marshalling.cpp

  LINK_LIBS
  clangdRemoteIndexProto
  clangDaemon
  clangdSupport

  DEPENDS
  clangdRemoteIndexProto
  )
