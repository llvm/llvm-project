add_entrypoint_object(
  getcontext
  SRCS
    getcontext.cpp
  HDRS
    ../getcontext.h
  DEPENDS
    libc.include.llvm-libc-types.ucontext_t
    libc.include.sys_syscall
    libc.src.__support.common
    libc.src.__support.macros.config
    libc.hdr.types.size_t
)

add_entrypoint_object(
  setcontext
  SRCS
    setcontext.cpp
  HDRS
    ../setcontext.h
  DEPENDS
    libc.include.llvm-libc-types.ucontext_t
    libc.include.sys_syscall
    libc.src.__support.common
    libc.src.__support.macros.config
    libc.hdr.types.size_t
)

