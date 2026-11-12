set(LLVM_LINK_COMPONENTS
  Support
  )
add_clang_executable(clangd-index-server-monitor
  Monitor.cpp

  DEPENDS
  clangdRemoteIndexServiceProto

  # Disable PCH, gRPC adds -pthread.
  DISABLE_PCH_REUSE
  )

target_link_libraries(clangd-index-server-monitor
  PRIVATE
  clangBasic
  clangdSupport

  clangdMonitoringServiceProto
  clangdRemoteIndexServiceProto
  )
