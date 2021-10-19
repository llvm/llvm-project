add_clang_library(clangdRemoteMarshalling
  Marshalling.cpp

  LINK_LIBS
  RemoteIndexProto

  clangDaemon
  clangdSupport

  DEPENDS
  RemoteIndexProto
  )
