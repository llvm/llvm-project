include(AddUnitTest)

# Make the targets default_gtest and default_gtest_main available.
build_gtest()

add_custom_target(LibsyclUnitTests)

set_target_properties(LibsyclUnitTests PROPERTIES FOLDER "libsycl/Tests/Unit")

add_subdirectory(mock)
add_subdirectory(device_selector)
add_subdirectory(event)
add_subdirectory(platform)
add_subdirectory(program_manager)
add_subdirectory(queue)

# The unit test executables are GoogleTest binaries that are discovered and run
# by lit, mirroring how liboffload runs its unit tests.
set(LIBSYCL_UNITTEST_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

add_lit_testsuite(check-sycl-unittests "Running libsycl unit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS LibsyclUnitTests libsycl-toolchain)
