if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

add_libc_test(
  vdso_test
  SUITE libc-osutil-tests
  SRCS vdso_test.cpp
  DEPENDS
    libc.src.__support.OSUtil.vdso
    libc.include.llvm-libc-types.struct_timeval # TODO: update this to proxy header once available
)
