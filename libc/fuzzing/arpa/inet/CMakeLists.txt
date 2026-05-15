add_libc_fuzzer(
  inet_aton_differential_fuzz
  SRCS
    inet_aton_differential_fuzz.cpp
  DEPENDS
    libc.src.arpa.inet.inet_aton
)
