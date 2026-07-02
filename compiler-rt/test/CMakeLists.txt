# Needed for lit support in standalone builds.
include(AddLLVM)

pythonize_bool(COMPILER_RT_TEST_STANDALONE_BUILD_LIBS)

pythonize_bool(LLVM_ENABLE_EXPENSIVE_CHECKS)

pythonize_bool(ZLIB_FOUND)
pythonize_bool(COMPILER_RT_BUILD_STANDALONE_LIBATOMIC)

pythonize_bool(COMPILER_RT_ENABLE_INTERNAL_SYMBOLIZER)

pythonize_bool(SANITIZER_DISABLE_SYMBOLIZER_PATH_SEARCH)

pythonize_bool(COMPILER_RT_HAS_AARCH64_SME)

pythonize_bool(COMPILER_RT_HAS_NO_DEFAULT_CONFIG_FLAG)

pythonize_bool(SANITIZER_AMDHSA)

if(LLVM_TREE_AVAILABLE OR NOT COMPILER_RT_STANDALONE_BUILD)
  set(COMPILER_RT_BUILT_WITH_LLVM TRUE)
else()
  set(COMPILER_RT_BUILT_WITH_LLVM FALSE)
endif()
pythonize_bool(COMPILER_RT_BUILT_WITH_LLVM)

configure_compiler_rt_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.common.configured.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.common.configured)

# BlocksRuntime (and most of builtins) testsuites are not yet ported to lit.
# add_subdirectory(BlocksRuntime)

set(SANITIZER_COMMON_LIT_TEST_DEPS)

if(COMPILER_RT_BUILD_PROFILE AND COMPILER_RT_HAS_PROFILE)
  list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS profile)
endif()

# When ANDROID, we build tests with the host compiler (i.e. CMAKE_C_COMPILER),
# and run tests with tools from the host toolchain.
if(NOT ANDROID)
  if(NOT COMPILER_RT_STANDALONE_BUILD AND NOT LLVM_RUNTIMES_BUILD)
    # Use LLVM utils and Clang from the same build tree.
    list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS
      clang clang-resource-headers FileCheck count not llvm-config llvm-nm 
      llvm-objdump llvm-readelf llvm-readobj llvm-size llvm-symbolizer 
      compiler-rt-headers sancov split-file llvm-strip)
    if (WIN32)
      list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS KillTheDoctor)
    endif()
  endif()
  # Tests use C++ standard library headers.
  if (TARGET cxx-headers OR HAVE_LIBCXX)
    list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS cxx-headers)
  endif()
endif()

umbrella_lit_testsuite_begin(check-compiler-rt)

set(COMPILER_RT_KNOWN_TEST_SUITES
    builtins;ctx_profile;fuzzer;interception;lsan;memprof;metadata
    ;orc;profile;sanitizer_common;shadowcallstack
    ;ubsan;xray)
list(APPEND COMPILER_RT_KNOWN_TEST_SUITES ${ALL_SANITIZERS})
list(REMOVE_DUPLICATES COMPILER_RT_KNOWN_TEST_SUITES)
# Sort the list so that's easier to read when emitting errors.
list(SORT COMPILER_RT_KNOWN_TEST_SUITES)

set(COMPILER_RT_ENABLE_TEST_SUITES "all" CACHE STRING
    "List of test suites to enable (semicolon-separated), or \"all\" to enable all test suites. Known test suites: ${COMPILER_RT_KNOWN_TEST_SUITES}")

# Validate COMPILER_RT_ENABLE_TEST_SUITES values.
if(NOT "${COMPILER_RT_ENABLE_TEST_SUITES}" STREQUAL "all")
  foreach(suite ${COMPILER_RT_ENABLE_TEST_SUITES})
    if(NOT "${suite}" IN_LIST COMPILER_RT_KNOWN_TEST_SUITES)
      message(FATAL_ERROR "Unknown test suite '${suite}' in COMPILER_RT_ENABLE_TEST_SUITES. Known test suites are: ${COMPILER_RT_KNOWN_TEST_SUITES}")
    endif()
  endforeach()
endif()

function(compiler_rt_test_runtime runtime)
  cmake_parse_arguments(ARG "NO_COMPILER_RT_HAS_GUARD" "" "" ${ARGN})

  # Validate that the test suite name is known.
  if(NOT "${runtime}" IN_LIST COMPILER_RT_KNOWN_TEST_SUITES)
    message(FATAL_ERROR "'${runtime}' is not a known test suite. Known test suites are: ${COMPILER_RT_KNOWN_TEST_SUITES}")
  endif()

  # If COMPILER_RT_ENABLE_TEST_SUITES is not "all", check that this runtime
  # is in the enabled list.
  if(NOT "${COMPILER_RT_ENABLE_TEST_SUITES}" STREQUAL "all")
    if(NOT "${runtime}" IN_LIST COMPILER_RT_ENABLE_TEST_SUITES)
      message(STATUS "Skipping compiler-rt ${runtime} test directory")
      return()
    endif()
  endif()

  # FIXME: The old code had this but we should probably remove it
  # because it's unreachable because `compiler-rt/test` is guarded
  # by `if(COMPILER_RT_INCLUDE_TESTS)`.
  if(NOT COMPILER_RT_INCLUDE_TESTS)
    message(STATUS "Skipping compiler-rt ${runtime} test directory")
    return()
  endif()

  if(NOT ARG_NO_COMPILER_RT_HAS_GUARD)
    string(TOUPPER ${runtime} runtime_uppercase)
    if(NOT COMPILER_RT_HAS_${runtime_uppercase})
      message(STATUS "Skipping compiler-rt ${runtime} test directory")
      return()
    endif()
  endif()

  if(${runtime} STREQUAL cfi AND NOT COMPILER_RT_HAS_UBSAN)
    # CFI tests require diagnostic mode, which is implemented in UBSan.
    message(STATUS "Skipping ${runtime} test directory")
  elseif(${runtime} STREQUAL scudo_standalone)
    message(STATUS "Adding scudo/standalone test directory")
    add_subdirectory(scudo/standalone)
  else()
    message(STATUS "Adding compiler-rt ${runtime} test directory")
    add_subdirectory(${runtime})
  endif()
endfunction()

# Run sanitizer tests only if we're sure that clang would produce
# working binaries.
if(COMPILER_RT_CAN_EXECUTE_TESTS)
  # COMPILER_RT_TEST_EXTERNAL_BUILTINS allows running tests against builtins built
  # in an independent build. This option is only indended to be used by
  # LLVM_ENABLE_RUNTIMES-based builds.
  if(COMPILER_RT_BUILD_BUILTINS OR COMPILER_RT_TEST_EXTERNAL_BUILTINS)
    compiler_rt_test_runtime(builtins NO_COMPILER_RT_HAS_GUARD)
  endif()
  if(COMPILER_RT_BUILD_SANITIZERS)
    compiler_rt_test_runtime(interception)

    compiler_rt_test_runtime(lsan)
    compiler_rt_test_runtime(ubsan)
    compiler_rt_test_runtime(sanitizer_common)

    # GNU/Hurd and OpenBSD not supporting asan, cannot run the tests
    if(COMPILER_RT_BUILD_LIBFUZZER AND NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "GNU" AND NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "OpenBSD" AND NOT ANDROID)
      compiler_rt_test_runtime(fuzzer)

      # These tests don't need an additional runtime but use asan runtime.
      compiler_rt_test_runtime(metadata NO_COMPILER_RT_HAS_GUARD)
    endif()

    foreach(sanitizer ${COMPILER_RT_SANITIZERS_TO_BUILD})
      compiler_rt_test_runtime(${sanitizer})
    endforeach()
  endif()
  if(COMPILER_RT_BUILD_PROFILE AND COMPILER_RT_HAS_PROFILE
     AND NOT COMPILER_RT_GPU_BUILD)
    compiler_rt_test_runtime(profile)
  endif()
  if(COMPILER_RT_BUILD_CTX_PROFILE)
    compiler_rt_test_runtime(ctx_profile)
  endif()
  if(COMPILER_RT_BUILD_MEMPROF)
    compiler_rt_test_runtime(memprof)
  endif()
  if(COMPILER_RT_BUILD_XRAY)
    compiler_rt_test_runtime(xray)
  endif()
  if(COMPILER_RT_BUILD_ORC)
    compiler_rt_test_runtime(orc)
  endif()
  # ShadowCallStack does not yet provide a runtime with compiler-rt, the tests
  # include their own minimal runtime
  compiler_rt_test_runtime(shadowcallstack NO_COMPILER_RT_HAS_GUARD)
endif()

# Now that we've traversed all the directories and know all the lit testsuites,
# introduce a rule to run to run all of them.
get_property(LLVM_COMPILER_RT_LIT_DEPENDS GLOBAL PROPERTY LLVM_COMPILER_RT_LIT_DEPENDS)
add_custom_target(compiler-rt-test-depends)
set_target_properties(compiler-rt-test-depends PROPERTIES FOLDER "Compiler-RT/Tests")
if(LLVM_COMPILER_RT_LIT_DEPENDS)
  add_dependencies(compiler-rt-test-depends ${LLVM_COMPILER_RT_LIT_DEPENDS})
endif()
umbrella_lit_testsuite_end(check-compiler-rt)

if(COMPILER_RT_STANDALONE_BUILD)
  if(NOT TARGET check-all)
    add_custom_target(check-all)
  endif()
  add_dependencies(check-all check-compiler-rt)
endif()
