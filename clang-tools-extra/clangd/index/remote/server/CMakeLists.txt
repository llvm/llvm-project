set(LLVM_LINK_COMPONENTS
  Support
  )
add_clang_executable(clangd-index-server
  Server.cpp

  DEPENDS
  clangdRemoteIndexProto
  clangdRemoteIndexServiceProto
  )

target_link_libraries(clangd-index-server
  PRIVATE
  clangDaemon
  clangdSupport

  clangdMonitoringServiceProto
  clangdRemoteIndexProto
  clangdRemoteIndexServiceProto
  clangdRemoteMarshalling

  ${REFLECTION_LIBRARY}
  )
