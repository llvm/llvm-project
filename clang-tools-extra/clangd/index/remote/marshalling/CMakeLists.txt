add_clang_library(clangdRemoteMarshalling
  Marshalling.cpp

  LINK_LIBS
  RemoteIndexProto

  protobuf
  clangDaemon
  clangdSupport

  DEPENDS
  RemoteIndexProto
  )
