function(add_unittest_framework_library name)
  cmake_parse_arguments(
    "TEST_LIB"
    "" # No optional arguments
    "" # No single value arguments
    "SRCS;HDRS;DEPENDS" # Multi value arguments
    ${ARGN}
  )
  if(NOT TEST_LIB_SRCS)
    message(FATAL_ERROR "'add_unittest_framework_library' requires SRCS; for "
                        "header only libraries, use 'add_header_library'")
  endif()

  foreach(lib IN ITEMS ${name}.unit ${name}.hermetic)
    add_library(
      ${lib}
      STATIC
      EXCLUDE_FROM_ALL
      ${TEST_LIB_SRCS}
      ${TEST_LIB_HDRS}
    )
    target_include_directories(${lib} PRIVATE ${LIBC_SOURCE_DIR})
    if(TARGET libc.src.time.clock)
      target_compile_definitions(${lib} PRIVATE TARGET_SUPPORTS_CLOCK)
    endif()
  endforeach()

  if(LLVM_LIBC_FULL_BUILD)
    # TODO: Build test framework with LIBC_FULL_BUILD in full build mode after
    # making LibcFPExceptionHelpers and LibcDeathTestExecutors hermetic.
    set(LLVM_LIBC_FULL_BUILD "")
    _get_common_test_compile_options(compile_options "" "")
    target_compile_options(${name}.unit PRIVATE ${compile_options})
    set(LLVM_LIBC_FULL_BUILD ON)
  else()
    _get_common_test_compile_options(compile_options "" "")
    target_compile_options(${name}.unit PRIVATE ${compile_options})
  endif()

  _get_hermetic_test_compile_options(compile_options "")
  target_include_directories(${name}.hermetic PRIVATE ${LIBC_INCLUDE_DIR})
  target_compile_options(${name}.hermetic PRIVATE ${compile_options} -nostdinc++)

  if(TEST_LIB_DEPENDS)
    foreach(dep IN ITEMS ${TEST_LIB_DEPENDS})
      if(TARGET ${dep}.unit)
        add_dependencies(${name}.unit ${dep}.unit)
      else()
        add_dependencies(${name}.unit ${dep})
      endif()
      if(TARGET ${dep}.hermetic)
        add_dependencies(${name}.hermetic ${dep}.hermetic)
      else()
        add_dependencies(${name}.hermetic ${dep})
      endif()
    endforeach()
  endif()
endfunction()

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
  DEPENDS
    libc.src.__support.big_int
    libc.src.__support.c_string
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.fixed_point.fx_rep
    libc.src.__support.macros.properties.types
    libc.src.__support.OSUtil.osutil
    libc.src.__support.uint128
)

set(libc_death_test_srcs LibcDeathTestExecutors.cpp)
if(${LIBC_TARGET_OS} STREQUAL "linux")
  list(APPEND libc_death_test_srcs ExecuteFunctionUnix.cpp)
endif()

add_unittest_framework_library(
  LibcDeathTestExecutors
  SRCS
    ${libc_death_test_srcs}
  HDRS
    ExecuteFunction.h
)

add_unittest_framework_library(
  LibcHermeticTestSupport
  SRCS
    HermeticTestUtils.cpp
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
  LibcFPTestHelpers
  SRCS
    FEnvSafeTest.cpp
    RoundingModeUtils.cpp
  HDRS
    FEnvSafeTest.h
    FPMatcher.h
    RoundingModeUtils.h
  DEPENDS
    LibcTest
    libc.test.UnitTest.string_utils
    libc.src.__support.CPP.array
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
    libc.src.__support.FPUtil.fenv_impl
    libc.src.__support.FPUtil.rounding_mode
)

add_unittest_framework_library(
  LibcFPExceptionHelpers
  SRCS
    FPExceptMatcher.cpp
  HDRS
    FPExceptMatcher.h
  DEPENDS
    LibcTest
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fenv_impl
    libc.hdr.types.fenv_t
)

add_unittest_framework_library(
  LibcMemoryHelpers
  SRCS
    MemoryMatcher.cpp
  HDRS
    MemoryMatcher.h
  DEPENDS
    LibcTest
    libc.src.__support.CPP.span
)

add_unittest_framework_library(
  LibcPrintfHelpers
  SRCS
    PrintfMatcher.cpp
  HDRS
    PrintfMatcher.h
  DEPENDS
    LibcTest
    libc.src.__support.FPUtil.fp_bits
    libc.src.stdio.printf_core.core_structs
    libc.test.UnitTest.string_utils
)

add_unittest_framework_library(
  LibcScanfHelpers
  SRCS
    ScanfMatcher.cpp
  HDRS
    ScanfMatcher.h
  DEPENDS
    LibcTest
    libc.src.__support.FPUtil.fp_bits
    libc.src.stdio.scanf_core.core_structs
    libc.test.UnitTest.string_utils
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
    libc.src.errno.errno
)
