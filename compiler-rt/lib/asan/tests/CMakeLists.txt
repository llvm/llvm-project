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

# Compile source for the given architecture, using compiler
# options in ${ARGN}, and add it to the object list.
macro(asan_compile obj_list source arch kind)
  get_filename_component(basename ${source} NAME)
  if(CMAKE_CONFIGURATION_TYPES)
    set(output_obj "${CMAKE_CFG_INTDIR}/${obj_list}.${basename}.${arch}${kind}.o")
  else()
    set(output_obj "${obj_list}.${basename}.${arch}${kind}.o")
  endif()
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  set(COMPILE_DEPS ${ASAN_UNITTEST_HEADERS} ${ASAN_BLACKLIST_FILE})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND COMPILE_DEPS gtest asan)
  endif()
  clang_compile(${output_obj} ${source}
                CFLAGS ${ARGN} ${TARGET_CFLAGS}
                DEPS ${COMPILE_DEPS})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

# Link ASan unit test for a given architecture from a set
# of objects in with given linker flags.
macro(add_asan_test test_suite test_name arch kind)
  cmake_parse_arguments(TEST "WITH_TEST_RUNTIME" "" "OBJECTS;LINK_FLAGS;SUBDIR" ${ARGN})
  get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
  set(TEST_DEPS ${TEST_OBJECTS})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND TEST_DEPS asan)
  endif()
  if(TEST_WITH_TEST_RUNTIME)
    list(APPEND TEST_DEPS ${ASAN_TEST_RUNTIME})
    if(CMAKE_CONFIGURATION_TYPES)
     set(configuration_path "${CMAKE_CFG_INTDIR}/")
    else()
     set(configuration_path "")
    endif()
    if(NOT MSVC)
      set(asan_test_runtime_path ${configuration_path}lib${ASAN_TEST_RUNTIME}.a)
    else()
      set(asan_test_runtime_path ${configuration_path}${ASAN_TEST_RUNTIME}.lib)
    endif()
    list(APPEND TEST_OBJECTS ${asan_test_runtime_path})
  endif()
  add_compiler_rt_test(${test_suite} ${test_name}
                       SUBDIR ${TEST_SUBDIR}
                       OBJECTS ${TEST_OBJECTS}
                       DEPS ${TEST_DEPS}
                       LINK_FLAGS ${TEST_LINK_FLAGS}
                                  ${TARGET_LINK_FLAGS})
endmacro()

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
  list(APPEND ASAN_INST_TEST_SOURCES asan_mac_test.cc)
endif()

set(ASAN_BENCHMARKS_SOURCES
  ${COMPILER_RT_GTEST_SOURCE}
  asan_benchmarks_test.cc)

# Adds ASan unit tests and benchmarks for architecture.
macro(add_asan_tests_for_arch_and_kind arch kind)
  # Instrumented tests.
  set(ASAN_INST_TEST_OBJECTS)
  foreach(src ${ASAN_INST_TEST_SOURCES})
    asan_compile(ASAN_INST_TEST_OBJECTS ${src} ${arch} ${kind}
      ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} ${ARGN})
  endforeach()
  if (APPLE)
    # Add Mac-specific helper.
    asan_compile(ASAN_INST_TEST_OBJECTS asan_mac_test_helpers.mm ${arch} ${kind}
                 ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} -ObjC ${ARGN})
  endif()

  if (MSVC)
    # With the MSVC CRT, the choice between static and dynamic CRT is made at
    # compile time with a macro. Simulate the effect of passing /MD to clang-cl.
    set(ASAN_INST_DYNAMIC_TEST_OBJECTS)
    foreach(src ${ASAN_INST_TEST_SOURCES})
      asan_compile(ASAN_INST_DYNAMIC_TEST_OBJECTS ${src} ${arch} ${kind}
        ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} -D_MT -D_DLL ${ARGN})
    endforeach()
    # Clang links the static CRT by default. Override that to use the dynamic
    # CRT.
    set(ASAN_DYNAMIC_UNITTEST_INSTRUMENTED_LINK_FLAGS
      ${ASAN_DYNAMIC_UNITTEST_INSTRUMENTED_LINK_FLAGS}
      -Wl,-nodefaultlib:libcmt,-defaultlib:msvcrt,-defaultlib:oldnames)
  else()
    set(ASAN_INST_DYNAMIC_TEST_OBJECTS ${ASAN_INST_TEST_OBJECTS})
  endif()

  # Create the 'default' folder where ASAN tests are produced.
  if(CMAKE_CONFIGURATION_TYPES)
    foreach(build_mode ${CMAKE_CONFIGURATION_TYPES})
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/default/${build_mode}")
    endforeach()
  else()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/default")
  endif()

  add_asan_test(AsanUnitTests "Asan-${arch}${kind}-Test"
                ${arch} ${kind} SUBDIR "default"
                OBJECTS ${ASAN_INST_TEST_OBJECTS}
                LINK_FLAGS ${ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS})
  if(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME)
    # Create the 'dynamic' folder where ASAN tests are produced.
    if(CMAKE_CONFIGURATION_TYPES)
      foreach(build_mode ${CMAKE_CONFIGURATION_TYPES})
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dynamic/${build_mode}")
      endforeach()
    else()
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dynamic")
    endif()

    add_asan_test(AsanDynamicUnitTests "Asan-${arch}${kind}-Dynamic-Test"
                  ${arch} ${kind} SUBDIR "dynamic"
                  OBJECTS ${ASAN_INST_DYNAMIC_TEST_OBJECTS}
                  LINK_FLAGS ${ASAN_DYNAMIC_UNITTEST_INSTRUMENTED_LINK_FLAGS})
  endif()

  # Add static ASan runtime that will be linked with uninstrumented tests.
  set(ASAN_TEST_RUNTIME RTAsanTest.${arch}${kind})
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
  # Uninstrumented tests.
  set(ASAN_NOINST_TEST_OBJECTS)
  foreach(src ${ASAN_NOINST_TEST_SOURCES})
    asan_compile(ASAN_NOINST_TEST_OBJECTS ${src} ${arch} ${kind}
                 ${ASAN_UNITTEST_COMMON_CFLAGS} ${ARGN})
  endforeach()
  add_asan_test(AsanUnitTests "Asan-${arch}${kind}-Noinst-Test"
                ${arch} ${kind} SUBDIR "default"
                OBJECTS ${ASAN_NOINST_TEST_OBJECTS}
                LINK_FLAGS ${ASAN_UNITTEST_NOINST_LINK_FLAGS}
                WITH_TEST_RUNTIME)

  # Benchmarks.
  set(ASAN_BENCHMARKS_OBJECTS)
  foreach(src ${ASAN_BENCHMARKS_SOURCES})
    asan_compile(ASAN_BENCHMARKS_OBJECTS ${src} ${arch} ${kind}
                 ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} ${ARGN})
  endforeach()
  add_asan_test(AsanBenchmarks "Asan-${arch}${kind}-Benchmark"
                ${arch} ${kind} SUBDIR "default"
                OBJECTS ${ASAN_BENCHMARKS_OBJECTS}
                LINK_FLAGS ${ASAN_UNITTEST_INSTRUMENTED_LINK_FLAGS})
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS AND NOT ANDROID)
  set(ASAN_TEST_ARCH ${ASAN_SUPPORTED_ARCH})
  if(APPLE)
    darwin_filter_host_archs(ASAN_SUPPORTED_ARCH ASAN_TEST_ARCH)
  endif()
  foreach(arch ${ASAN_TEST_ARCH})
    add_asan_tests_for_arch_and_kind(${arch} "-inline")
    add_asan_tests_for_arch_and_kind(${arch} "-with-calls"
      -mllvm -asan-instrumentation-with-call-threshold=0)
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
