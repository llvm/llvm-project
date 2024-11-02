# Needed for lit support in standalone builds.
include(AddLLVM)

option(COMPILER_RT_TEST_STANDALONE_BUILD_LIBS
  "When set to ON and testing in a standalone build, test the runtime \
  libraries built by this standalone build rather than the runtime libraries \
  shipped with the compiler (used for testing). When set to OFF and testing \
  in a standalone build, test the runtime libraries shipped with the compiler \
  (used for testing). This option has no effect if the compiler and this \
  build are configured to use the same runtime library path."
  ON)
pythonize_bool(COMPILER_RT_TEST_STANDALONE_BUILD_LIBS)

pythonize_bool(LLVM_ENABLE_EXPENSIVE_CHECKS)
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
      clang clang-resource-headers FileCheck count not llvm-config llvm-nm llvm-objdump
      llvm-readelf llvm-readobj llvm-size llvm-symbolizer compiler-rt-headers sancov
      llvm-jitlink)
    if (WIN32)
      list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS KillTheDoctor)
    endif()
  endif()
  # Tests use C++ standard library headers.
  if (TARGET cxx-headers OR HAVE_LIBCXX)
    list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS cxx-headers)
  endif()
endif()

if(COMPILER_RT_STANDALONE_BUILD)
  umbrella_lit_testsuite_begin(check-compiler-rt)
endif()

function(compiler_rt_test_runtime runtime)
  string(TOUPPER ${runtime} runtime_uppercase)
  if(COMPILER_RT_HAS_${runtime_uppercase})
    add_subdirectory(${runtime})
    foreach(directory ${ARGN})
      add_subdirectory(${directory})
    endforeach()
  endif()
endfunction()

# Run sanitizer tests only if we're sure that clang would produce
# working binaries.
if(COMPILER_RT_CAN_EXECUTE_TESTS)
  if(COMPILER_RT_BUILD_BUILTINS)
    add_subdirectory(builtins)
  endif()
  if(COMPILER_RT_BUILD_SANITIZERS)
    compiler_rt_test_runtime(interception)

    compiler_rt_test_runtime(lsan)
    # CFI tests require diagnostic mode, which is implemented in UBSan.
    compiler_rt_test_runtime(ubsan cfi)
    compiler_rt_test_runtime(sanitizer_common)

    # OpenBSD not supporting asan, cannot run the tests
    if(COMPILER_RT_BUILD_LIBFUZZER AND NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "OpenBSD" AND NOT ANDROID)
      compiler_rt_test_runtime(fuzzer)
    endif()

    foreach(sanitizer ${COMPILER_RT_SANITIZERS_TO_BUILD})
      # cfi testing is gated on ubsan
      if(NOT ${sanitizer} STREQUAL cfi)
        compiler_rt_test_runtime(${sanitizer})
      endif()
    endforeach()
  endif()
  if(COMPILER_RT_BUILD_PROFILE AND COMPILER_RT_HAS_PROFILE)
    compiler_rt_test_runtime(profile)
  endif()
  if(COMPILER_RT_BUILD_MEMPROF)
    compiler_rt_test_runtime(memprof)
  endif()
  if(COMPILER_RT_BUILD_XRAY)
    compiler_rt_test_runtime(xray)
  endif()
  if(COMPILER_RT_BUILD_ORC)
    compiler_rt_Test_runtime(orc)
  endif()
  if(COMPILER_RT_BUILD_CRT)
    add_subdirectory(crt)
  endif()
  # ShadowCallStack does not yet provide a runtime with compiler-rt, the tests
  # include their own minimal runtime
  add_subdirectory(shadowcallstack)
endif()

if(COMPILER_RT_STANDALONE_BUILD)
  # Now that we've traversed all the directories and know all the lit testsuites,
  # introduce a rule to run to run all of them.
  add_custom_target(compiler-rt-test-depends DEPENDS ${LLVM_COMPILER_RT_LIT_DEPENDS})
  umbrella_lit_testsuite_end(check-compiler-rt)

  if(NOT TARGET check-all)
    add_custom_target(check-all)
  endif()
  add_dependencies(check-all check-compiler-rt)
endif()
