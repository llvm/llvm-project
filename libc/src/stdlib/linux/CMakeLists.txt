add_header_library(
  abort_utils
  HDRS
    abort_utils.h
  DEPENDS
    libc.src.__support.OSUtil.linux.syscall_wrappers.raise
    libc.src.__support.OSUtil.osutil
    libc.src.signal.linux.__restore
    libc.src.signal.linux.signal_utils
)

add_entrypoint_object(
  setenv
  SRCS
    setenv.cpp
  HDRS
    ../setenv.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.__support.common
    libc.src.__support.libc_errno
    libc.src.__support.macros.config
    libc.src.__support.macros.null_check
    libc.src.stdlib.environ_internal
)

add_entrypoint_object(
  unsetenv
  SRCS
    unsetenv.cpp
  HDRS
    ../unsetenv.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.__support.common
    libc.src.__support.libc_errno
    libc.src.__support.macros.config
    libc.src.__support.macros.null_check
    libc.src.stdlib.environ_internal
)

add_header_library(
  vdso_rng
  HDRS
    vdso_rng.h
  DEPENDS
    libc.src.__support.threads.thread # For __cxa_thread_atexit_impl
    libc.src.__support.CPP.algorithm
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.mutex
    libc.src.__support.CPP.optional
    libc.src.__support.OSUtil.linux.vdso
    libc.src.__support.OSUtil.osutil
    libc.src.__support.macros.config
    libc.src.__support.threads.linux.raw_mutex
    libc.src.sys.auxv.getauxval # TODO: remove public entrypoint dependency
    libc.include.sys_syscall
)
