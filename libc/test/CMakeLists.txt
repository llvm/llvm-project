add_custom_target(libc-unit-tests)
add_custom_target(libc-hermetic-tests)

add_custom_target(exhaustive-check-libc)
add_custom_target(libc-long-running-tests)

# Build-only targets for lit (don't run tests, just build executables)
add_custom_target(libc-unit-tests-build)
add_custom_target(libc-hermetic-tests-build)
add_custom_target(libc-integration-tests-build)
add_custom_target(libc_include_tests-build)

# Check if tests can safely include signal.h. (FPExceptMatcher.cpp can
# use it, but can also do without it.)
if(NOT LLVM_LIBC_FULL_BUILD)
  # In an overlay build of libc, assume signal.h is available: the host libc
  # will provide it even if we don't.
  add_compile_definitions(HAVE_SIGNAL_H)
else()
  # In a full build, signal.h only exists if we provide it ourselves.
  # In a bare-metal full build we might not, so check which headers
  # we're shipping.
  list(FIND TARGET_PUBLIC_HEADERS libc.include.signal SIGNAL_H_idx)
  if(SIGNAL_H_idx GREATER -1)
    add_compile_definitions(HAVE_SIGNAL_H)
  endif()
endif()

# Configure the site config file for lit
configure_lit_site_cfg(
  ${LIBC_SOURCE_DIR}/test/lit.site.cfg.py.in
  ${LIBC_BUILD_DIR}/test/lit.site.cfg.py
  MAIN_CONFIG
  ${LIBC_SOURCE_DIR}/test/lit.cfg.py
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "LLVM_LIBS_DIR"
  "LIBC_SOURCE_DIR"
  "LIBC_BUILD_DIR"
  "CMAKE_CROSSCOMPILING_EMULATOR"
)

add_custom_target(check-libc-build)
add_lit_testsuite(check-libc
  "Running libc tests via lit"
  ${LIBC_BUILD_DIR}/test
)
add_dependencies(check-libc check-libc-build)

if(TARGET check-hdrgen)
  add_dependencies(check-libc check-hdrgen)
endif()

add_dependencies(check-libc-build libc-unit-tests-build)
add_dependencies(check-libc-build libc-hermetic-tests-build)

add_subdirectory(UnitTest)

if(LIBC_TARGET_OS_IS_GPU)
  if(NOT CMAKE_CROSSCOMPILING_EMULATOR)
    message(WARNING "Cannot build libc GPU tests, set CMAKE_CROSSCOMPILING_EMULATOR.")
    return()
  elseif(LIBC_GPU_TESTS_DISABLED)
    message(WARNING "Cannot build libc GPU tests, missing target architecture.")
    return()
  endif()
endif()

add_subdirectory(src)
add_subdirectory(utils)
add_subdirectory(shared)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_subdirectory(include)

if(NOT ${LIBC_TARGET_OS} STREQUAL "linux" AND
  NOT ${LIBC_TARGET_OS} STREQUAL "gpu")
  # Integration tests are currently only available for linux and the GPU.
  return()
endif()

add_subdirectory(IntegrationTest)
add_subdirectory(integration)
