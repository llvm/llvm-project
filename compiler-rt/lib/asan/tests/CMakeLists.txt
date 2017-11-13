# Testing rules for AddressSanitizer.
#
# These are broken into two buckets. One set of tests directly interacts with
# the runtime library and checks its functionality. These are the
# no-instrumentation tests.
#
# Another group of tests relies upon the ability to compile the test with
# address sanitizer instrumentation pass. These tests form "integration" tests
# and have some elements of version skew -- they test the *host* compiler's
# instrumentation against the just-built runtime library.

include(CheckCXXCompilerFlag)
include(CompilerRTCompile)

include_directories(..)
include_directories(../..)

set(ASAN_UNITTEST_HEADERS
  asan_mac_test.h
  asan_test_config.h
  asan_test_utils.h)

set(ASAN_UNITTEST_COMMON_CFLAGS
  ${COMPILER_RT_UNITTEST_CFLAGS}
  ${COMPILER_RT_GTEST_CFLAGS}
  ${COMPILER_RT_ASAN_SHADOW_SCALE_LLVM_FLAG}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/asan
  -I${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common/tests
  -fno-rtti
  -O2
  -Wno-format
  -Werror=sign-compare
  -Wno-non-virtual-dtor)
append_list_if(COMPILER_RT_HAS_WVARIADIC_MACROS_FLAG -Wno-variadic-macros ASAN_UNITTEST_COMMON_CFLAGS)

# This will ensure the target linker is used
# during cross compilation
set(ASAN_UNITTEST_COMMON_LINK_FLAGS
  ${COMPILER_RT_UNITTEST_LINK_FLAGS})

# -gline-tables-only must be enough for ASan, so use it if possible.
if(COMPILER_RT_TEST_COMPILER_ID MATCHES "Clang")
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS -gline-tables-only)
else()
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS -g)
endif()
if(MSVC)
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS -gcodeview)
endif()
list(APPEND ASAN_UNITTEST_COMMON_LINK_FLAGS -g)

# Use -D instead of definitions to please custom compile command.
list(APPEND ASAN_UNITTEST_COMMON_CFLAGS
  ${COMPILER_RT_ASAN_SHADOW_SCALE_FLAG}
  -DASAN_HAS_BLACKLIST=1
  -DASAN_HAS_EXCEPTIONS=1
  -DASAN_UAR=0)

if(APPLE)
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS ${DARWIN_osx_CFLAGS})
  list(APPEND ASAN_UNITTEST_COMMON_LINK_FLAGS ${DARWIN_osx_LINK_FLAGS})

  add_weak_symbols("asan" WEAK_SYMBOL_LINK_FLAGS)
  add_weak_symbols("ubsan" WEAK_SYMBOL_LINK_FLAGS)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)
  list(APPEND ASAN_UNITTEST_COMMON_LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS})
endif()

if(MSVC)
  # Disable exceptions on Windows until they work reliably.
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS -fno-exceptions -DGTEST_HAS_SEH=0)
endif()

set(ASAN_BLACKLIST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/asan_test.ignore")
set(ASAN_UNITTEST_INSTRUMENTED_CFLAGS
  ${ASAN_UNITTEST_COMMON_CFLAGS}
  -fsanitize=address
  "-fsanitize-blacklist=${ASAN_BLACKLIST_FILE}"
)
if(CAN_TARGET_x86_64 OR CAN_TARGET_i386)
  list(APPEND ASAN_UNITTEST_INSTRUMENTED_CFLAGS -mllvm -asan-instrument-assembly)
endif()

if(NOT MSVC)
  list(APPEND ASAN_UNITTEST_COMMON_LINK_FLAGS --driver-mode=g++)
endif()

# x86_64 FreeBSD 9.2 additionally requires libc++ to build the tests.
if(CMAKE_SYSTEM MATCHES "FreeBSD-9.2-RELEASE")
  list(APPEND ASAN_UNITTEST_COMMON_LINK_FLAGS "-lc++")
endif()

# Unit tests on Mac depend on Foundation.
if(APPLE)
  list(APPEND ASAN_UNITTEST_COMMON_LINK_FLAGS -framework Foundation)
endif()
if(ANDROID)
  list(APPEND ASAN_UNITTEST_COMMON_LINK_FLAGS -pie)
endif()

set(ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS
  ${ASAN_UNITTEST_COMMON_LINK_FLAGS})
list(APPEND ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS -fsanitize=address)

set(ASAN_DYNAMIC_UNITTEST_INSTRUMENTED_LINK_FLAGS
  ${ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS}
  -shared-libasan)

set(ASAN_UNITTEST_INSTRUMENTED_LIBS)
# NDK r10 requires -latomic almost always.
append_list_if(ANDROID atomic ASAN_UNITTEST_INSTRUMENTED_LIBS)

set(ASAN_UNITTEST_NOINST_LINK_FLAGS ${ASAN_UNITTEST_COMMON_LINK_FLAGS})
if(NOT APPLE)
  append_list_if(COMPILER_RT_HAS_LIBM -lm ASAN_UNITTEST_NOINST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBDL -ldl ASAN_UNITTEST_NOINST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBRT -lrt ASAN_UNITTEST_NOINST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBPTHREAD -pthread ASAN_UNITTEST_NOINST_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LIBPTHREAD -pthread ASAN_DYNAMIC_UNITTEST_INSTRUMENTED_LINK_FLAGS)
endif()

# TODO(eugenis): move all -l flags above to _LIBS?
set(ASAN_UNITTEST_NOINST_LIBS)
append_list_if(COMPILER_RT_HAS_LIBLOG log ASAN_UNITTEST_NOINST_LIBS)
# NDK r10 requires -latomic almost always.
append_list_if(ANDROID atomic ASAN_UNITTEST_NOINST_LIBS)

# Main AddressSanitizer unit tests.
add_custom_target(AsanUnitTests)
set_target_properties(AsanUnitTests PROPERTIES FOLDER "Compiler-RT Tests")

# AddressSanitizer unit tests with dynamic runtime (on platforms where it's
# not the default).
add_custom_target(AsanDynamicUnitTests)
set_target_properties(AsanDynamicUnitTests PROPERTIES FOLDER "Compiler-RT Tests")
# ASan benchmarks (not actively used now).
add_custom_target(AsanBenchmarks)
set_target_properties(AsanBenchmarks PROPERTIES FOLDER "Compiler-RT Tests")

set(ASAN_NOINST_TEST_SOURCES
  ${COMPILER_RT_GTEST_SOURCE}
  asan_fake_stack_test.cc
  asan_noinst_test.cc
  asan_test_main.cc)

set(ASAN_INST_TEST_SOURCES
  ${COMPILER_RT_GTEST_SOURCE}
  asan_asm_test.cc
  asan_globals_test.cc
  asan_interface_test.cc
  asan_internal_interface_test.cc
  asan_test.cc
  asan_oob_test.cc
  asan_mem_test.cc
  asan_str_test.cc
  asan_test_main.cc)
if(APPLE)
  list(APPEND ASAN_INST_TEST_SOURCES asan_mac_test.cc asan_mac_test_helpers.mm)
endif()

set(ASAN_BENCHMARKS_SOURCES
  ${COMPILER_RT_GTEST_SOURCE}
  asan_benchmarks_test.cc)

function(add_asan_tests arch test_runtime)
  cmake_parse_arguments(TEST "" "KIND" "CFLAGS" ${ARGN})

  # Closure to keep the values.
  function(generate_asan_tests test_objects test_suite testname)
    generate_compiler_rt_tests(${test_objects} ${test_suite} ${testname} ${arch}
      COMPILE_DEPS ${ASAN_UNITTEST_HEADERS} ${ASAN_BLACKLIST_FILE}
      DEPS gtest asan
      KIND ${TEST_KIND}
      ${ARGN}
      )
    set("${test_objects}" "${${test_objects}}" PARENT_SCOPE)
  endfunction()

  set(ASAN_INST_TEST_OBJECTS)
  generate_asan_tests(ASAN_INST_TEST_OBJECTS AsanUnitTests
    "Asan-${arch}${TEST_KIND}-Test"
    SUBDIR "default"
    LINK_FLAGS ${ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS}
    SOURCES ${ASAN_INST_TEST_SOURCES}
    CFLAGS ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} ${TEST_CFLAGS})

  if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
    set(dynamic_test_name "Asan-${arch}${TEST_KIND}-Dynamic-Test")
    if(MSVC)

      # With the MSVC CRT, the choice between static and dynamic CRT is made at
      # compile time with a macro. Simulate the effect of passing /MD to clang-cl.
      set(ASAN_DYNAMIC_TEST_OBJECTS)
      generate_asan_tests(ASAN_DYNAMIC_TEST_OBJECTS
        AsanDynamicUnitTests "${dynamic_test_name}"
        SUBDIR "dynamic"
        CFLAGS ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} -D_MT -D_DLL
        SOURCES ${ASAN_INST_TEST_SOURCES}
        LINK_FLAGS ${ASAN_DYNAMIC_UNITTEST_INSTRUMENTED_LINK_FLAGS}
          -Wl,-nodefaultlib:libcmt,-defaultlib:msvcrt,-defaultlib:oldnames
        )
    else()

      # Otherwise, reuse ASAN_INST_TEST_OBJECTS.
      add_compiler_rt_test(AsanDynamicUnitTests "${dynamic_test_name}" "${arch}"
        SUBDIR "dynamic"
        OBJECTS ${ASAN_INST_TEST_OBJECTS}
        DEPS asan ${ASAN_INST_TEST_OBJECTS}
        LINK_FLAGS ${ASAN_DYNAMIC_UNITTEST_INSTRUMENTED_LINK_FLAGS}
        )
    endif()
  endif()

  # Uninstrumented tests.
  set(ASAN_NOINST_TEST_OBJECTS)
  generate_asan_tests(ASAN_NOINST_TEST_OBJECTS
    AsanUnitTests "Asan-${arch}${TEST_KIND}-Noinst-Test"
    SUBDIR "default"
    CFLAGS ${ASAN_UNITTEST_COMMON_CFLAGS}
    LINK_FLAGS ${ASAN_UNITTEST_NOINST_LINK_FLAGS}
    SOURCES ${ASAN_NOINST_TEST_SOURCES}
    RUNTIME ${test_runtime})

  set(ASAN_BENCHMARK_OBJECTS)
  generate_asan_tests(ASAN_BENCHMARK_OBJECTS
    AsanBenchmarks "Asan-${arch}${TEST_KIND}-Benchmark"
    SUBDIR "default"
    CFLAGS ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS}
    SOURCES ${ASAN_BENCHMARKS_SOURCES}
    LINK_FLAGS ${ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS})
endfunction()

if(COMPILER_RT_CAN_EXECUTE_TESTS AND NOT ANDROID)
  set(ASAN_TEST_ARCH ${ASAN_SUPPORTED_ARCH})
  if(APPLE)
    darwin_filter_host_archs(ASAN_SUPPORTED_ARCH ASAN_TEST_ARCH)
  endif()

  foreach(arch ${ASAN_TEST_ARCH})

    # Add static ASan runtime that will be linked with uninstrumented tests.
    set(ASAN_TEST_RUNTIME RTAsanTest.${arch})
    if(APPLE)
      set(ASAN_TEST_RUNTIME_OBJECTS
        $<TARGET_OBJECTS:RTAsan_dynamic.osx>
        $<TARGET_OBJECTS:RTInterception.osx>
        $<TARGET_OBJECTS:RTSanitizerCommon.osx>
        $<TARGET_OBJECTS:RTSanitizerCommonLibc.osx>
        $<TARGET_OBJECTS:RTLSanCommon.osx>
        $<TARGET_OBJECTS:RTUbsan.osx>)
    else()
      set(ASAN_TEST_RUNTIME_OBJECTS
        $<TARGET_OBJECTS:RTAsan.${arch}>
        $<TARGET_OBJECTS:RTAsan_cxx.${arch}>
        $<TARGET_OBJECTS:RTInterception.${arch}>
        $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
        $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
        $<TARGET_OBJECTS:RTLSanCommon.${arch}>
        $<TARGET_OBJECTS:RTUbsan.${arch}>
        $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>)
    endif()
    add_library(${ASAN_TEST_RUNTIME} STATIC ${ASAN_TEST_RUNTIME_OBJECTS})
    set_target_properties(${ASAN_TEST_RUNTIME} PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      FOLDER "Compiler-RT Runtime tests")

    add_asan_tests(${arch} ${ASAN_TEST_RUNTIME} KIND "-inline")
    add_asan_tests(${arch} ${ASAN_TEST_RUNTIME} KIND "-calls"
      CFLAGS -mllvm -asan-instrumentation-with-call-threshold=0)
  endforeach()
endif()

if(ANDROID)
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    # Test w/o ASan instrumentation. Link it with ASan statically.
    add_executable(AsanNoinstTest # FIXME: .arch?
      $<TARGET_OBJECTS:RTAsan.${arch}>
      $<TARGET_OBJECTS:RTInterception.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
      $<TARGET_OBJECTS:RTUbsan.${arch}>
      $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>
      ${COMPILER_RT_GTEST_SOURCE}
      ${ASAN_NOINST_TEST_SOURCES})
    set_target_compile_flags(AsanNoinstTest ${ASAN_UNITTEST_COMMON_CFLAGS})
    set_target_link_flags(AsanNoinstTest ${ASAN_UNITTEST_NOINST_LINK_FLAGS})
    target_link_libraries(AsanNoinstTest ${ASAN_UNITTEST_NOINST_LIBS})

    # Test with ASan instrumentation. Link with ASan dynamic runtime.
    add_executable(AsanTest
      ${COMPILER_RT_GTEST_SOURCE}
      ${ASAN_INST_TEST_SOURCES})
    set_target_compile_flags(AsanTest ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS})
    set_target_link_flags(AsanTest ${ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS})
    target_link_libraries(AsanTest ${ASAN_UNITTEST_INSTRUMENTED_LIBS})

    # Setup correct output directory and link flags.
    set_target_properties(AsanNoinstTest AsanTest PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # Add unit tests to the test suite.
    add_dependencies(AsanUnitTests AsanNoinstTest AsanTest)
  endforeach()
endif()
