add_clang_library(clangdRemoteMarshalling
  Marshalling.cpp

  LINK_LIBS
  clangdRemoteIndexProto
  clangDaemon
  clangdSupport

  DEPENDS
  clangdRemoteIndexProto
  )
