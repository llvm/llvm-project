set(LLVM_LINK_COMPONENTS
  LineEditor
  Support
  )
add_clang_executable(clangd-index-client
  Client.cpp
  )
target_compile_definitions(clangd-index-client PRIVATE -DGOOGLE_PROTOBUF_NO_RTTI=1)
clang_target_link_libraries(clangd-index-client
  PRIVATE
  clangDaemon
  )
target_link_libraries(clangd-index-client
  PRIVATE
  RemoteIndexProtos

  protobuf
  grpc++
  )
