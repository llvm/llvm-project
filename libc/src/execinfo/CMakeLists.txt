add_header_library(
  unwind
  HDRS
    unwind.h
  DEPENDS
    libc.hdr.stdint_proxy
)

add_header_library(
  backtrace_line
  HDRS
    backtrace_line.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.hdr.types.size_t
)

add_entrypoint_object(
  backtrace
  SRCS
    backtrace.cpp
  HDRS
    backtrace.h
  COMPILE_OPTIONS
    ${libc_opt_high_flag}
    # The unwinder walks the stack from the frame this call is in, so this one
    # translation unit needs the tables to describe it. The rest of the
    # library is built without them.
    -fasynchronous-unwind-tables
  DEPENDS
    .unwind
    libc.src.__support.threads.callonce
    libc.src.dlfcn.dlopen
    libc.src.dlfcn.dlsym
)

add_entrypoint_object(
  backtrace_symbols_fd
  SRCS
    backtrace_symbols_fd.cpp
  HDRS
    backtrace_symbols_fd.h
  DEPENDS
    .backtrace_line
    libc.hdr.types.dl_info
    libc.src.__support.OSUtil.linux.syscall_wrappers.write
    libc.src.dlfcn.dladdr
)

add_entrypoint_object(
  backtrace_symbols
  SRCS
    backtrace_symbols.cpp
  HDRS
    backtrace_symbols.h
  DEPENDS
    .backtrace_line
    libc.hdr.types.dl_info
    libc.hdr.types.size_t
    libc.src.dlfcn.dladdr
)
