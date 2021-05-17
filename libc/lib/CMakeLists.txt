add_entrypoint_library(
  llvmlibc
  DEPENDS
  ${TARGET_LLVMLIBC_ENTRYPOINTS}
)

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE})
else()
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX})
endif()

install(
  TARGETS llvmlibc
  ARCHIVE DESTINATION ${LIBC_INSTALL_PREFIX}/${LIBC_INSTALL_LIBRARY_DIR}
  COMPONENT llvmlibc
)

add_custom_target(
    install-llvmlibc
    DEPENDS llvmlibc
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=llvmlibc
            -P "${LIBC_BINARY_DIR}/cmake_install.cmake"
)
