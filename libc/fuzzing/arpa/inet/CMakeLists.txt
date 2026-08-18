add_libc_fuzzer(
  inet_aton_differential_fuzz
  SRCS
    inet_aton_differential_fuzz.cpp
  DEPENDS
    libc.src.__support.CPP.scope
    libc.src.arpa.inet.inet_aton
    libc.src.string.memcpy
)

add_libc_fuzzer(
  inet_ntop_differential_fuzz
  SRCS
    inet_ntop_differential_fuzz.cpp
  DEPENDS
    libc.src.arpa.inet.inet_ntop
    libc.src.errno.errno
)
