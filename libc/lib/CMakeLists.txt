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
  ARCHIVE DESTINATION "${LIBC_INSTALL_LIBRARY_DIR}"
  COMPONENT llvmlibc
)

add_llvm_install_targets(
  install-llvmlibc
  DEPENDS llvmlibc
  COMPONENT llvmlibc
)
