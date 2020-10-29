set(LLVM_LINK_COMPONENTS
  Support
  )
add_clang_executable(clangd-index-server
  Server.cpp

  DEPENDS
  RemoteIndexProto
  RemoteIndexServiceProto
  )

target_link_libraries(clangd-index-server
  PRIVATE
  clangDaemon
  clangdSupport

  RemoteIndexProto
  RemoteIndexServiceProto
  clangdRemoteMarshalling

  grpc++
  )
