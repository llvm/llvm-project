add_library(llvmlibc_rpc_server STATIC rpc_server.cpp)

# Include the RPC implemenation from libc.
target_include_directories(llvmlibc_rpc_server PRIVATE ${LIBC_SOURCE_DIR})
target_include_directories(llvmlibc_rpc_server PUBLIC ${LIBC_SOURCE_DIR}/include)
target_include_directories(llvmlibc_rpc_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# Ignore unsupported clang attributes if we're using GCC.
target_compile_options(llvmlibc_rpc_server PUBLIC
                       $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>)
target_compile_definitions(llvmlibc_rpc_server PUBLIC
                           LIBC_NAMESPACE=${LIBC_NAMESPACE})

# This utility needs to be compiled for the host system when cross compiling.
if(LLVM_RUNTIMES_TARGET OR LIBC_TARGET_TRIPLE)
  target_compile_options(llvmlibc_rpc_server PUBLIC
                         --target=${LLVM_HOST_TRIPLE})
  target_link_libraries(llvmlibc_rpc_server PUBLIC
                        "--target=${LLVM_HOST_TRIPLE}")
endif()

# Install the server and associated header.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/llvmlibc_rpc_server.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT libc-headers)
install(TARGETS llvmlibc_rpc_server
        ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}"
        COMPONENT libc)
