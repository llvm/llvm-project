set(LLVM_LINK_COMPONENTS
  Support
  )
add_clang_executable(clangd-index-server
  Server.cpp

  DEPENDS
  RemoteIndexProtos
  )

target_link_libraries(clangd-index-server
  PRIVATE
  clangDaemon
  clangdSupport

  RemoteIndexProtos
  clangdRemoteMarshalling

  grpc++
  )
