set(LLVM_LINK_COMPONENTS
  Support
  )
add_clang_executable(clangd-index-server-monitor
  Monitor.cpp

  DEPENDS
  RemoteIndexServiceProto
  )

target_link_libraries(clangd-index-server-monitor
  PRIVATE
  clangBasic
  clangdSupport

  MonitoringServiceProto
  RemoteIndexServiceProto
  )
