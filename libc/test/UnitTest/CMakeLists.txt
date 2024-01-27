# When the clock target is available, it will be used inside the test lib
# to indicate the performance of the tests.
set(clock_target "")
if(libc.src.time.clock IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS)
  set(clock_target
      libc.src.time.clock
      libc.include.llvm-libc-macros.time_macros)
endif()

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

  # The Nvidia 'nvlink' linker does not support static libraries.
  if(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
    set(library_type OBJECT)
  else()
    set(library_type STATIC)
  endif()

  foreach(lib IN ITEMS ${name}.unit ${name}.hermetic)
    add_library(
      ${lib}
      ${library_type}
      EXCLUDE_FROM_ALL
      ${TEST_LIB_SRCS}
      ${TEST_LIB_HDRS}
    )
    set_target_properties(${lib} PROPERTIES
      TEST_FRAMEWORK TRUE)
    target_include_directories(${lib} PUBLIC ${LIBC_SOURCE_DIR})
    target_compile_options(${lib} PRIVATE -fno-exceptions -fno-rtti)
    if(clock_target)
      message(STATUS "   Clock targets: ${clock_target}")
      target_compile_definitions(${lib} PRIVATE LIBC_TEST_USE_CLOCK)
    endif()
  endforeach()

  target_include_directories(${name}.hermetic PRIVATE ${LIBC_BUILD_DIR}/include)
  target_compile_options(${name}.hermetic
      PRIVATE ${LIBC_HERMETIC_TEST_COMPILE_OPTIONS} -ffreestanding -nostdinc++)

  if(TEST_LIB_DEPENDS)
    set(normal_deps "")
    set(framework_deps "")
    foreach(dep IN LISTS TEST_LIB_DEPENDS)
      if(TARGET ${dep})
        get_target_property(dep_is_framework ${dep} TEST_FRAMEWORK)
        if(dep_is_framework)
          list(APPEND framework_deps ${dep})
        else()
          list(APPEND normal_deps ${dep})
        endif()
      else()
        list(APPEND normal_deps ${dep})
      endif()
    endforeach()

    foreach(dep IN LISTS framework_deps)
      target_link_libraries(${name}.unit ${dep}.unit)
      target_link_libraries(${name}.hermetic ${dep}.hermetic)
    endforeach()

    get_object_files_for_test(
        link_object_files skipped_entrypoints_list
        INTERNAL TRUE
        ${normal_deps}
    )

    target_link_libraries(${name}.unit ${link_object_files})
    if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
      target_link_libraries(${name}.hermetic ${link_object_files})
    else()
      get_object_files_for_test(
          link_object_files skipped_entrypoints_list
          INTERNAL FALSE
          ${normal_deps}
      )
      target_link_libraries(${name}.hermetic ${link_object_files})
    endif()
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
    libc.src.__support.c_string
    libc.src.__support.CPP.string
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.OSUtil.osutil
    libc.src.__support.uint128
    ${clock_target}
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
    libc.src.__support.CPP.string
    libc.src.__support.CPP.type_traits
)

add_unittest_framework_library(
  LibcFPTestHelpers
  SRCS
    RoundingModeUtils.cpp
  HDRS
    FPMatcher.h
    RoundingModeUtils.h
  DEPENDS
    LibcTest
    libc.test.UnitTest.string_utils
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
    libc.src.__support.FPUtil.fenv_impl
    libc.src.__support.FPUtil.rounding_mode
    libc.src.errno.errno
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
