add_libc_exhaustive_testsuite(libc-mathvec-exhaustivetests)

add_header_library(
  exhaustive_test
  HDRS
    exhaustive_test.h
  DEPENDS
    libc.src.__support.CPP.simd
)

add_fp_unittest(
  expf_test
  NO_RUN_POSTBUILD
  SUITE
    libc-mathvec-exhaustivetests
  SRCS
    expf_test.cpp
  DEPENDS
    .exhaustive_test
    libc.src.math.expf
    libc.src.mathvec.expf
    libc.src.__support.CPP.simd
  LINK_LIBRARIES
    -lpthread
)
