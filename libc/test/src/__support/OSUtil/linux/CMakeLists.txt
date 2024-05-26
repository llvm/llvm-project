if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

add_libc_test(
  vdso_test
  SUITE libc-osutil-tests
  SRCS vdso_test.cpp
  DEPENDS
    libc.src.__support.OSUtil.linux.vdso
    libc.hdr.types.struct_timeval
    libc.hdr.types.struct_timespec
    libc.hdr.types.clockid_t
    libc.hdr.time_macros
)
