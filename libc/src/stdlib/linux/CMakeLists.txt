add_entrypoint_object(
  abort
  SRCS
    abort.cpp
  HDRS
    ../abort.h
  DEPENDS
    libc.include.stdlib
    libc.src.signal.raise
    libc.src.stdlib._Exit
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
    libc.src.__support.threads.callonce
    libc.src.__support.threads.linux.raw_mutex
    libc.src.sys.auxv.getauxval # TODO: remove public entrypoint dependency
    libc.include.sys_syscall
)
