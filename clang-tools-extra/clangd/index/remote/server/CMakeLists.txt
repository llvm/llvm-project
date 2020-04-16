set(LLVM_LINK_COMPONENTS
  LineEditor
  Support
  )
add_clang_executable(clangd-index-server
  Server.cpp
  )
target_compile_definitions(clangd-index-server PRIVATE -DGOOGLE_PROTOBUF_NO_RTTI=1)
clang_target_link_libraries(clangd-index-server
  PRIVATE
  clangDaemon
  )
target_link_libraries(clangd-index-server
  PRIVATE
  RemoteIndexProtos

  protobuf
  grpc++
  clangDaemon
  )
