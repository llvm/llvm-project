add_custom_target(libc_include_tests)
add_dependencies(check-libc libc_include_tests)

add_libc_test(
  assert_test
  SUITE
    libc_include_tests
  SRCS
    assert_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.assert_macros
)

add_libc_test(
  complex_test
  SUITE
    libc_include_tests
  SRCS
    complex_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.complex_macros
)

add_libc_test(
  sys_queue_test
  SUITE
    libc_include_tests
  SRCS
    sys/queue_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.sys_queue_macros
    libc.src.__support.char_vector
    libc.src.__support.CPP.string
  COMPILE_OPTIONS
    # This is needed because the __containerof macro uses statement expression.
    -Wno-gnu-statement-expression-from-macro-expansion
)

add_libc_test(
  stdbit_test
  SUITE
    libc_include_tests
  HDRS
    stdbit_stub.h
  SRCS
    stdbit_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.stdbit_macros
    libc.include.llvm_libc_common_h
    libc.include.stdbit
    # Intentionally do not depend on libc.src.stdbit.*. The include test is
    # simply testing the macros provided by stdbit.h, not the implementation
    # of the underlying functions which the type generic macros may dispatch
    # to.
)
add_libc_test(
  stdbit_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  HDRS
    stdbit_stub.h
  SRCS
    stdbit_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.stdbit_macros
    libc.include.llvm_libc_common_h
    libc.include.stdbit
    libc.src.assert.__assert_fail
    # Intentionally do not depend on libc.src.stdbit.*. The include test is
    # simply testing the macros provided by stdbit.h, not the implementation
    # of the underlying functions which the type generic macros may dispatch
    # to.
)

add_libc_test(
  stdckdint_test
  SUITE
    libc_include_tests
  SRCS
    stdckdint_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.stdckdint_macros
)

add_libc_test(
  issubnormal_test
  SUITE
    libc_include_tests
  SRCS
    issubnormal_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  issubnormalf_test
  SUITE
    libc_include_tests
  SRCS
    issubnormalf_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  issubnormall_test
  SUITE
    libc_include_tests
  SRCS
    issubnormall_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnormal_test
  SUITE
    libc_include_tests
  SRCS
    isnormal_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnormalf_test
  SUITE
    libc_include_tests
  SRCS
    isnormalf_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnormall_test
  SUITE
    libc_include_tests
  SRCS
    isnormall_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  fpclassify_test
  SUITE
    libc_include_tests
  SRCS
    fpclassify_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  fpclassifyf_test
  SUITE
    libc_include_tests
  SRCS
    fpclassifyf_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  fpclassifyl_test
  SUITE
    libc_include_tests
  SRCS
    fpclassifyl_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  iszero_test
  SUITE
    libc_include_tests
  SRCS
    iszero_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  iszerof_test
  SUITE
    libc_include_tests
  SRCS
    iszerof_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  iszerol_test
  SUITE
    libc_include_tests
  SRCS
    iszerol_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  netinet_in_test
  SUITE
    libc_include_tests
  SRCS
    netinet_in_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.netinet_in_macros
)

add_libc_test(
  signbit_test
  SUITE
    libc_include_tests
  SRCS
    signbit_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  signbitf_test
  SUITE
    libc_include_tests
  SRCS
    signbitf_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  signbitl_test
  SUITE
    libc_include_tests
  SRCS
    signbitl_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnan_test
  SUITE
    libc_include_tests
  SRCS
    isnan_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnanf_test
  SUITE
    libc_include_tests
  SRCS
    isnanf_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnanl_test
  SUITE
    libc_include_tests
  SRCS
    isnanl_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isinf_test
  SUITE
    libc_include_tests
  SRCS
    isinf_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isinff_test
  SUITE
    libc_include_tests
  SRCS
    isinff_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isinfl_test
  SUITE
    libc_include_tests
  SRCS
    isinfl_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isfinite_test
  SUITE
    libc_include_tests
  SRCS
    isfinite_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isfinitef_test
  SUITE
    libc_include_tests
  SRCS
    isfinitef_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isfinitel_test
  SUITE
    libc_include_tests
  SRCS
    isfinitel_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  signbit_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    signbit_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnan_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    isnan_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isnormal_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    isnormal_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

# TODO(https://github.com/llvm/llvm-project/issues/114618): fix linkage failures
# add_libc_test(
#   issignaling_c_test
#   C_TEST
#   UNIT_TEST_ONLY
#   SUITE
#     libc_include_tests
#   SRCS
#     issignaling_test.c
#   COMPILE_OPTIONS
#     -Wall
#     -Werror
#   DEPENDS
#     libc.include.llvm-libc-macros.math_function_macros
#     libc.src.math.issignaling
#     libc.src.math.issignalingf
#     libc.src.math.issignalingl
# )

# add_libc_test(
#   iscanonical_c_test
#   C_TEST
#   UNIT_TEST_ONLY
#   SUITE
#     libc_include_tests
#   SRCS
#     iscanonical_test.c
#   COMPILE_OPTIONS
#     -Wall
#     -Werror
#   DEPENDS
#     libc.include.llvm-libc-macros.math_function_macros
#     libc.src.math.iscanonical
#     libc.src.math.iscanonicalf
#     libc.src.math.iscanonicall
# )

add_libc_test(
  isinf_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    isinf_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  isfinite_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    isfinite_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  issubnormal_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    issubnormal_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  fpclassify_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    fpclassify_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  iszero_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    iszero_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_function_macros
)

add_libc_test(
  math_constants_c_test
  C_TEST
  UNIT_TEST_ONLY
  SUITE
    libc_include_tests
  SRCS
    math_constants_test.c
  COMPILE_OPTIONS
    -Wall
    -Werror
  DEPENDS
    libc.include.llvm-libc-macros.math_macros
)

# Test `#include <...>` of each header in each available language mode.
# This is gated on -DLLVM_LIBC_BUILD_HEADER_TESTS=ON until all the bugs
# in headers are fixed so the tests all compile.
set(TEST_STDC_VERSIONS 89;99;11;17;23)
set(TEST_STDCXX_VERSIONS 03;11;14;17;20;23;26)

function(add_header_test target_name source_file deps std_mode)
  if(LLVM_LIBC_BUILD_HEADER_TESTS)
    add_libc_test(
      ${target_name}
      C_TEST
      HERMETIC_TEST_ONLY
      SUITE
	libc_include_tests
      SRCS
	${source_file}
      COMPILE_OPTIONS
	-Werror
	-Wsystem-headers
	-Wall
	-Wextra
	-std=${std_mode}
      DEPENDS
	${deps}
    )
  endif()
endfunction()

foreach(target ${TARGET_PUBLIC_HEADERS})
  string(REPLACE "libc.include." "" header ${target})
  get_target_property(HEADER_NAME ${target} HEADER_NAME)

  set(test_stdc_file "${CMAKE_CURRENT_BINARY_DIR}/${header}_test.c")
  configure_file(header-test-template.c ${test_stdc_file} @ONLY)
  foreach(stdc_version ${TEST_STDC_VERSIONS})
    add_header_test(
      "${header}_c${stdc_version}_test"
      ${test_stdc_file}
      ${target}
      "c${stdc_version}"
    )
    add_header_test(
      "${header}_gnu${stdc_version}_test"
      ${test_stdc_file}
      ${target}
      "gnu${stdc_version}"
    )
  endforeach()

  set(test_stdcxx_file "${CMAKE_CURRENT_BINARY_DIR}/${header}_test.cpp")
  configure_file(header-test-template.c ${test_stdcxx_file} @ONLY)
  foreach(stdcxx_version ${TEST_STDCXX_VERSIONS})
    add_header_test(
      "${header}_cpp${stdcxx_version}_test"
      ${test_stdcxx_file}
      ${target}
      "c++${stdcxx_version}"
    )
    add_header_test(
      "${header}_gnucpp${stdcxx_version}_test"
      ${test_stdcxx_file}
      ${target}
      "gnu++${stdcxx_version}"
    )
  endforeach()
endforeach()
