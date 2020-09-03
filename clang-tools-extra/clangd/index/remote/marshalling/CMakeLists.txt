add_clang_library(clangdRemoteMarshalling
  Marshalling.cpp

  LINK_LIBS
  RemoteIndexProtos

  protobuf
  clangDaemon
  clangdSupport

  DEPENDS
  RemoteIndexProtos
  )
