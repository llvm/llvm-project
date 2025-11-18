add_entrypoint_object(
  catopen
  SRCS
    catopen.cpp
  HDRS
    catopen.h
  DEPENDS
    libc.include.llvm-libc-types.nl_catd
    libc.src.errno.errno
)

add_entrypoint_object(
  catclose
  SRCS
    catclose.cpp
  HDRS
    catclose.h
  DEPENDS
    libc.include.llvm-libc-types.nl_catd
)

add_entrypoint_object(
  catgets
  SRCS
    catgets.cpp
  HDRS
    catgets.h
  DEPENDS
    libc.include.llvm-libc-types.nl_catd
)

