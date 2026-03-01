add_custom_target(check-libc)
add_custom_target(libc-unit-tests)
add_custom_target(libc-hermetic-tests)
add_dependencies(check-libc libc-unit-tests libc-hermetic-tests)

add_custom_target(exhaustive-check-libc)
add_custom_target(libc-long-running-tests)

# Build-only targets for lit (don't run tests, just build executables)
add_custom_target(libc-unit-tests-build)
add_custom_target(libc-hermetic-tests-build)

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
)

add_lit_testsuite(check-libc-lit
  "Running libc tests via lit"
  ${LIBC_BUILD_DIR}/test
  DEPENDS libc-unit-tests-build libc-hermetic-tests-build
)

add_subdirectory(UnitTest)

if(LIBC_TARGET_OS_IS_GPU)
  if(NOT TARGET libc.utils.gpu.loader)
    message(WARNING "Cannot build libc GPU tests, missing loader.")
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
