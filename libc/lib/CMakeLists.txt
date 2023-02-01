add_entrypoint_library(
  libc
  DEPENDS
  ${TARGET_LLVMLIBC_ENTRYPOINTS}
)
set_target_properties(
  libc
  PROPERTIES
    ARCHIVE_OUTPUT_NAME ${LIBC_ARCHIVE_NAME}
)

if(LIBC_TARGET_TRIPLE)
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LIBC_TARGET_TRIPLE})
elseif(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT LIBC_GPU_BUILD)
  set(LIBC_INSTALL_LIBRARY_DIR
      lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE})
else()
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX})
endif()

install(
  TARGETS libc
  ARCHIVE DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
  COMPONENT libc-static-archives
)

add_llvm_install_targets(
  install-libc-static-archives
  DEPENDS libc
  COMPONENT libc-static-archives
)
