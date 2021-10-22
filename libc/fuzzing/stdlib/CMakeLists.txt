add_libc_fuzzer(
  qsort_fuzz
  SRCS
    qsort_fuzz.cpp
  DEPENDS
    libc.src.stdlib.qsort
)

add_libc_fuzzer(
  atof_fuzz
  SRCS
    atof_fuzz.cpp
  HDRS
    StringParserOutputDiff.h
  DEPENDS
    libc.src.stdlib.atof
)

