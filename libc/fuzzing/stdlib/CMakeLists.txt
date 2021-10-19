add_libc_fuzzer(
  qsort_fuzz
  SRCS
    qsort_fuzz.cpp
  DEPENDS
    libc.src.stdlib.qsort
)

