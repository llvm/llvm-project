if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  return()
endif()

add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
add_subdirectory(syscall_wrappers)

add_object_library(
  darwin_util
  SRCS
    exit.cpp
  HDRS
    io.h
    syscall.h
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.darwin_${LIBC_TARGET_ARCHITECTURE}_util
    libc.src.__support.common
    libc.src.__support.CPP.string_view
    libc.include.sys_syscall
)

if(LLVM_LIBC_FULL_BUILD)
  add_object_library(
    assert_fail
    SRCS
      assert_fail.cpp
    DEPENDS
      .darwin_util
      libc.src.__support.OSUtil.assert_fail_hdrs
      libc.src.__support.OSUtil.assert_fail_utils
      libc.src.__support.macros.config
  )
endif()
