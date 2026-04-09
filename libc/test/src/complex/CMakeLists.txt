add_custom_target(libc-complex-unittests)

add_fp_unittest(
  conj_test
  SUITE
    libc-complex-unittests
  SRCS
    conj_test.cpp
  DEPENDS
    libc.src.complex.conj
)

add_fp_unittest(
  conjf_test
  SUITE
    libc-complex-unittests
  SRCS
    conjf_test.cpp
  DEPENDS
    libc.src.complex.conjf
)

add_fp_unittest(
  conjl_test
  SUITE
    libc-complex-unittests
  SRCS
    conjl_test.cpp
  DEPENDS
    libc.src.complex.conjl
)

add_fp_unittest(
  conjf16_test
  SUITE
    libc-complex-unittests
  SRCS
    conjf16_test.cpp
  DEPENDS
    libc.src.complex.conjf16
)

add_fp_unittest(
  conjf128_test
  SUITE
    libc-complex-unittests
  SRCS
    conjf128_test.cpp
  DEPENDS
    libc.src.complex.conjf128
)

add_fp_unittest(
  cproj_test
  SUITE
    libc-complex-unittests
  SRCS
    cproj_test.cpp
  DEPENDS
    libc.src.complex.cproj
)

add_fp_unittest(
  cprojf_test
  NEED_MPC
  SUITE
    libc-complex-unittests
  SRCS
    cprojf_test.cpp
  DEPENDS
    libc.src.complex.cprojf
)

add_fp_unittest(
  cprojl_test
  SUITE
    libc-complex-unittests
  SRCS
    cprojl_test.cpp
  DEPENDS
    libc.src.complex.cprojl
)

add_fp_unittest(
  cprojf16_test
  SUITE
    libc-complex-unittests
  SRCS
    cprojf16_test.cpp
  DEPENDS
    libc.src.complex.cprojf16
)

add_fp_unittest(
  cprojf128_test
  SUITE
    libc-complex-unittests
  SRCS
    cprojf128_test.cpp
  DEPENDS
    libc.src.complex.cprojf128
)

add_fp_unittest(
  creal_test
  SUITE
    libc-complex-unittests
  SRCS
    creal_test.cpp
  DEPENDS
    libc.src.complex.creal
)

add_fp_unittest(
  crealf_test
  SUITE
    libc-complex-unittests
  SRCS
    crealf_test.cpp
  DEPENDS
    libc.src.complex.crealf
)

add_fp_unittest(
  creall_test
  SUITE
    libc-complex-unittests
  SRCS
    creall_test.cpp
  DEPENDS
    libc.src.complex.creall
)

add_fp_unittest(
  crealf16_test
  SUITE
    libc-complex-unittests
  SRCS
    crealf16_test.cpp
  DEPENDS
    libc.src.complex.crealf16
)

add_fp_unittest(
  crealf128_test
  SUITE
    libc-complex-unittests
  SRCS
    crealf128_test.cpp
  DEPENDS
    libc.src.complex.crealf128
)

add_fp_unittest(
  cimag_test
  SUITE
    libc-complex-unittests
  SRCS
    cimag_test.cpp
  DEPENDS
    libc.src.complex.cimag
)

add_fp_unittest(
  cimagf_test
  SUITE
    libc-complex-unittests
  SRCS
    cimagf_test.cpp
  DEPENDS
    libc.src.complex.cimagf
)

add_fp_unittest(
  cimagl_test
  SUITE
    libc-complex-unittests
  SRCS
    cimagl_test.cpp
  DEPENDS
    libc.src.complex.cimagl
)

add_fp_unittest(
  cimagf16_test
  SUITE
    libc-complex-unittests
  SRCS
    cimagf16_test.cpp
  DEPENDS
    libc.src.complex.cimagf16
)

add_fp_unittest(
  cimagf128_test
  SUITE
    libc-complex-unittests
  SRCS
    cimagf128_test.cpp
  DEPENDS
    libc.src.complex.cimagf128
)
