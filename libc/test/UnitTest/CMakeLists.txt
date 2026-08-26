function(add_unittest_framework_library target_name)
  add_target_with_flags(
    ${target_name}
    CREATE_TARGET _create_unittest_framework_library
    ${ARGN}
  )
endfunction()

function(_create_unittest_framework_library fq_target_name)
  cmake_parse_arguments(
    "TEST_LIB"
    "" # No optional arguments
    "" # No single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;FLAGS" # Multi value arguments
    ${ARGN}
  )
  if(NOT TEST_LIB_SRCS)
    message(FATAL_ERROR "'add_unittest_framework_library' requires SRCS; for "
      "header only libraries, use 'add_header_library'")
  endif()

  get_fq_deps_list(fq_deps_list ${TEST_LIB_DEPENDS})

  add_library(
    ${fq_target_name}
    OBJECT
    EXCLUDE_FROM_ALL
      ${TEST_LIB_SRCS}
      ${TEST_LIB_HDRS}
  )
  target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  if(TARGET libc.src.time.clock)
    target_compile_definitions(${fq_target_name} PRIVATE TARGET_SUPPORTS_CLOCK)
  endif()

  # Find out if the target supports catching signals.
  if(NOT LLVM_LIBC_FULL_BUILD)
    # In an overlay build of libc, assume signal.h is available: the host
    # libc will provide it even if we don't.
    set(no_signal FALSE)
  elseif(TARGET libc.src.signal.signal)
    # In a full build, signal.h only exists if we provide it ourselves, which
    # we might not on bare metal.
    get_target_property(no_signal libc.src.signal.signal SKIPPED)
  else()
    set(no_signal TRUE)
  endif()
  if(NOT no_signal)
    target_compile_definitions(${fq_target_name} PRIVATE TARGET_SUPPORTS_SIGNAL_CATCHING)
  endif()

  if(LLVM_LIBC_FULL_BUILD)
    _get_hermetic_test_compile_options(compile_options "" "${TEST_LIB_FLAGS}")
    target_compile_options(${fq_target_name} PRIVATE ${compile_options} -nostdinc++)
  else()
    _get_common_test_compile_options(compile_options "" "${TEST_LIB_FLAGS}")

    if(TEST_LIB_COMPILE_OPTIONS)
      list(APPEND compile_options ${TEST_LIB_COMPILE_OPTIONS})
    endif()
    target_compile_options(${fq_target_name} PRIVATE ${compile_options})
  endif()

  if(fq_deps_list)
    add_dependencies(${fq_target_name} ${fq_deps_list})
    target_link_libraries(${fq_target_name} PUBLIC ${fq_deps_list})
  endif()
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      DEPS "${fq_deps_list}"
      FLAGS "${ADD_OBJECT_FLAGS}"
      TARGET_TYPE ${OBJECT_LIBRARY_TARGET_TYPE}
      OBJECT_FILES "$<TARGET_OBJECTS:${fq_target_name}>"
  )
endfunction()

if(NOT TARGET libc.src.__support.OSUtil.osutil OR NOT LLVM_LIBC_FULL_BUILD)
  message(STATUS "TestLogger will use system libc's stdio to print.")
  set(test_logger_compile_options "-DLIBC_TEST_USE_SYSTEM_PRINTF")
  set(test_logger_osutil "")
else()
  set(test_logger_compile_options "")
  set(test_logger_osutil "libc.src.__support.OSUtil.osutil")
endif()

add_unittest_framework_library(
  LibcTest
  SRCS
    CmakeFilePath.cpp
    LibcTest.cpp
    LibcTestMain.cpp
    TestLogger.cpp
  HDRS
    LibcTest.h
    Test.h
    TestLogger.h
  COMPILE_OPTIONS
    ${test_logger_compile_options}
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.big_int
    libc.src.__support.c_string
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.macros.properties.compiler
    libc.src.__support.macros.properties.types
    libc.src.__support.wchar.string_converter
    libc.src.__support.uint128
    ${test_logger_osutil}
)

add_unittest_framework_library(
  LibcCTest
  SRCS
    LibcCTest.cpp
  HDRS
    LibcCTest.h
  DEPENDS
    .LibcTest
)

set(libc_hermetic_test_support_srcs HermeticTestUtils.cpp)
if(LIBC_TARGET_OS_IS_BAREMETAL AND LIBC_TARGET_ARCHITECTURE_IS_ARM)
  list(APPEND libc_hermetic_test_support_srcs ArmBaremetalTestUtils.cpp)
endif()

if(LIBC_TEST_SUBPROCESS_TESTS)
  set(libc_death_test_deps
    libc.hdr.stdint_proxy
    libc.src.__support.libc_assert
  )
  if(LLVM_LIBC_FULL_BUILD)
    list(APPEND libc_death_test_deps
      libc.include.llvm-libc-macros.poll-macros
      libc.include.llvm-libc-macros.signal_macros
      libc.include.llvm-libc-macros.sys_wait_macros
      libc.include.llvm-libc-types.pid_t
      libc.include.llvm-libc-types.struct_pollfd
    )
  endif()
  add_unittest_framework_library(
    LibcDeathTestExecutors
    SRCS
      ExecuteFunctionUnix.cpp
      LibcDeathTestExecutors.cpp
    HDRS
      ExecuteFunction.h
    DEPENDS
      ${libc_death_test_deps}
  )
else()
  add_header_library(
    LibcDeathTestExecutors
    HDRS
      ExecuteFunction.h
    DEPENDS
      libc.hdr.stdint_proxy
  )
endif()

add_unittest_framework_library(
  HermeticTestUtils
  SRCS
    ${libc_hermetic_test_support_srcs}
  DEPENDS
    libc.hdr.stdint_proxy
)

add_header_library(
  string_utils
  HDRS
    StringUtils.h
  DEPENDS
    libc.src.__support.big_int
    libc.src.__support.CPP.string
    libc.src.__support.CPP.type_traits
)

add_unittest_framework_library(
  FPTestHelpers
  SRCS
    FEnvSafeTest.cpp
    RoundingModeUtils.cpp
  HDRS
    FEnvSafeTest.h
    FPMatcher.h
    RoundingModeUtils.h
  DEPENDS
    .LibcTest
    .ErrnoCheckingTest
    .string_utils
    libc.src.__support.CPP.array
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
    libc.src.__support.FPUtil.fenv_impl
    libc.src.__support.FPUtil.rounding_mode
)

add_unittest_framework_library(
  FPExceptMatcher
  SRCS
    FPExceptMatcher.cpp
  HDRS
    FPExceptMatcher.h
  DEPENDS
    .LibcTest
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fenv_impl
    libc.hdr.types.fenv_t
)

add_unittest_framework_library(
  MemoryMatcher
  SRCS
    MemoryMatcher.cpp
  HDRS
    MemoryMatcher.h
  DEPENDS
    .LibcTest
    libc.src.__support.CPP.span
)

add_unittest_framework_library(
  PrintfMatcher
  SRCS
    PrintfMatcher.cpp
  HDRS
    PrintfMatcher.h
  DEPENDS
    .LibcTest
    .string_utils
    libc.hdr.stdint_proxy
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.printf_core.core_structs
    libc.src.__support.printf_core.printf_config
)

add_unittest_framework_library(
  ScanfMatcher
  SRCS
    ScanfMatcher.cpp
  HDRS
    ScanfMatcher.h
  DEPENDS
    .LibcTest
    .string_utils
    libc.hdr.stdint_proxy
    libc.src.__support.FPUtil.fp_bits
    libc.src.stdio.scanf_core.core_structs
)

add_header_library(
  ErrnoSetterMatcher
  HDRS
    ErrnoSetterMatcher.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.StringUtil.error_to_string
    libc.src.errno.errno
)

add_header_library(
  ErrnoCheckingTest
  HDRS
    ErrnoCheckingTest.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.macros.properties.architectures
    libc.src.errno.errno
)

add_header_library(
  ConstraintHandlerCheckingTest
  HDRS
    ConstraintHandlerCheckingTest.h
  DEPENDS
    libc.hdr.types.constraint_handler_t
    libc.hdr.types.errno_t
    libc.src.stdlib.set_constraint_handler_s
)
