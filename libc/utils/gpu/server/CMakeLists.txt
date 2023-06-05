add_library(rpc_server STATIC Server.cpp)

# Include the RPC implemenation from libc.
add_dependencies(rpc_server libc.src.__support.RPC.rpc)
target_include_directories(rpc_server PRIVATE ${LIBC_SOURCE_DIR})
target_include_directories(rpc_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
