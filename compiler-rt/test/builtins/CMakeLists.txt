set(BUILTINS_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(BUILTINS_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} builtins)
set(BUILTINS_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/TestCases)

# Test cases.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

#Unit tests.

include(builtin-config-ix)

# Indicate if this is an MSVC environment.
pythonize_bool(MSVC)

# Indicate if the compiler for the builtins library was MSVC. If the builtins
# compiler was clang-cl, we will enable some features that the host compiler
# will not, like C99 _Complex and int128.
set(BUILTINS_IS_MSVC OFF)
if (MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  set(BUILTINS_IS_MSVC ON)
endif()
pythonize_bool(BUILTINS_IS_MSVC)

#TODO: Add support for Apple.
if (NOT APPLE)
foreach(arch ${BUILTIN_SUPPORTED_ARCH})
  set(BUILTINS_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" BUILTINS_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} BUILTINS_TEST_TARGET_CC BUILTINS_TEST_TARGET_CFLAGS)
  if (${arch} STREQUAL "armhf")
    list(APPEND BUILTINS_TEST_TARGET_CFLAGS -fomit-frame-pointer -DCOMPILER_RT_ARMHF_TARGET)
    string(REPLACE ";" " " BUILTINS_TEST_TARGET_CFLAGS "${BUILTINS_TEST_TARGET_CFLAGS}")
  endif()

  if (${arch} STREQUAL "riscv32")
    list(APPEND BUILTINS_TEST_TARGET_CFLAGS -fforce-enable-int128)
    string(REPLACE ";" " " BUILTINS_TEST_TARGET_CFLAGS "${BUILTINS_TEST_TARGET_CFLAGS}")
  endif()

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py
    )
  list(APPEND BUILTINS_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
endforeach()
endif()

add_lit_testsuite(check-builtins "Running the Builtins tests"
  ${BUILTINS_TESTSUITES}
  DEPENDS ${BUILTINS_TEST_DEPS})
set_target_properties(check-builtins PROPERTIES FOLDER "Compiler-RT Misc")
