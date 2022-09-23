include_directories($<TARGET_PROPERTY:llvmRemoteCacheKVProto,BINARY_DIR>)
include_directories($<TARGET_PROPERTY:llvmRemoteCacheCASProto,BINARY_DIR>)

# Create an object library instead of a whole static archive.
add_library(clangRemoteCache OBJECT EXCLUDE_FROM_ALL
  Client.cpp
)
llvm_update_compile_flags(clangRemoteCache)

target_link_libraries(clangRemoteCache
  llvmRemoteCacheCASProto
  llvmRemoteCacheKVProto
)
add_dependencies(clangRemoteCache
  llvmRemoteCacheCASProto
  llvmRemoteCacheKVProto
)
