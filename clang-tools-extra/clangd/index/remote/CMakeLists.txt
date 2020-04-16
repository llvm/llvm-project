generate_grpc_protos(RemoteIndexProtos "Index.proto")

include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)

add_subdirectory(client)
add_subdirectory(server)
