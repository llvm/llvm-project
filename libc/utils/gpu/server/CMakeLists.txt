add_library(llvmlibc_rpc_server STATIC
  ${LIBC_SOURCE_DIR}/src/stdio/printf_core/writer.cpp
  ${LIBC_SOURCE_DIR}/src/stdio/printf_core/converter.cpp
  rpc_server.cpp
)

# Include the RPC implemenation from libc.
target_include_directories(llvmlibc_rpc_server PRIVATE ${LIBC_SOURCE_DIR})
target_include_directories(llvmlibc_rpc_server PUBLIC ${LIBC_SOURCE_DIR}/include)
target_include_directories(llvmlibc_rpc_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Ignore unsupported clang attributes if we're using GCC.
target_compile_options(llvmlibc_rpc_server PUBLIC
                       $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>)
target_compile_definitions(llvmlibc_rpc_server PUBLIC
                           LIBC_COPT_USE_C_ASSERT
                           LIBC_COPT_MEMCPY_USE_EMBEDDED_TINY
                           LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64
                           LIBC_COPT_ARRAY_ARG_LIST
                           LIBC_COPT_PRINTF_DISABLE_WRITE_INT
                           LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
                           LIBC_COPT_PRINTF_DISABLE_STRERROR
                           LIBC_NAMESPACE=${LIBC_NAMESPACE})

# Install the server and associated header.
install(FILES ${LIBC_SOURCE_DIR}/shared/rpc.h
              ${LIBC_SOURCE_DIR}/shared/rpc_util.h
              ${LIBC_SOURCE_DIR}/shared/rpc_opcodes.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shared
        COMPONENT libc-headers)
install(TARGETS llvmlibc_rpc_server
        ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}"
        COMPONENT libc)
