add_clang_library(clangdRemoteMarshalling STATIC
  Marshalling.cpp

  LINK_LIBS
  clangdRemoteIndexProto
  clangDaemon
  clangdSupport

  DEPENDS
  clangdRemoteIndexProto

  # Disable PCH, gRPC adds -pthread.
  DISABLE_PCH_REUSE
  )
