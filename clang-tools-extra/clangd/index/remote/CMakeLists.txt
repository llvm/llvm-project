if (CLANGD_ENABLE_REMOTE)
  generate_protos(clangdRemoteIndexProto "Index.proto")
  generate_protos(clangdMonitoringServiceProto "MonitoringService.proto"
    GRPC)
  generate_protos(clangdRemoteIndexServiceProto "Service.proto"
    DEPENDS "Index.proto"
    GRPC)
  # FIXME: Move this into generate_protos. Currently we only mention proto
  # filename as a dependency, but linking requires target name.
  target_link_libraries(clangdRemoteIndexServiceProto
    PRIVATE
    clangdRemoteIndexProto
    clangdMonitoringServiceProto
    )
  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  # FIXME(kirillbobyrev): target_compile_definitions is not working with
  # add_clang_library for some reason. Is there any way to make this
  # target-local?
  add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI=1)

  add_clang_library(clangdRemoteIndex
    Client.cpp

    LINK_LIBS
    clangdRemoteIndexProto
    clangdRemoteIndexServiceProto
    clangdRemoteMarshalling
    clangBasic
    clangDaemon
    clangdSupport

    DEPENDS
    clangdRemoteIndexProto
    clangdRemoteIndexServiceProto
    )

  add_subdirectory(marshalling)
  add_subdirectory(server)
  add_subdirectory(monitor)
else()
  # Provides a no-op implementation of clangdRemoteIndex.
  add_subdirectory(unimplemented)
endif()
