add_custom_target(libc-mathvec-unittests)

add_fp_unittest(
  expf_test
  SUITE
    libc-mathvec-unittests
  SRCS
    expf_test.cpp
  DEPENDS
    libc.src.math.expf
    libc.src.mathvec.expf
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.CPP.simd
  FLAGS
    ROUND_OPT
    FMA_OPT
)
