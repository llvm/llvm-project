add_entrypoint_object(
  init
  SRCS
    init.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.common
)

add_entrypoint_object(
  fini
  SRCS
    fini.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.common
)
