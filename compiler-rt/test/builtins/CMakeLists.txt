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

set(BUILTIN_TEST_ARCH ${BUILTIN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(BUILTIN_SUPPORTED_ARCH BUILTIN_TEST_ARCH)
endif()

foreach(arch ${BUILTIN_TEST_ARCH})
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

  if (${arch} MATCHES "arm|aarch64|arm64" AND COMPILER_RT_HAS_FLOAT16)
    list(APPEND BUILTINS_TEST_TARGET_CFLAGS -DCOMPILER_RT_HAS_FLOAT16)
    string(REPLACE ";" " " BUILTINS_TEST_TARGET_CFLAGS "${BUILTINS_TEST_TARGET_CFLAGS}")
  endif()

  # Compute builtins available in library and add them as lit features.
  if(APPLE)
    # TODO: Support other Apple platforms.
    set(BUILTIN_LIB_TARGET_NAME "clang_rt.builtins_${arch}_osx")
  else()
    set(BUILTIN_LIB_TARGET_NAME "clang_rt.builtins-${arch}")
  endif()
  if (NOT TARGET "${BUILTIN_LIB_TARGET_NAME}")
    message(FATAL_ERROR "Target ${BUILTIN_LIB_TARGET_NAME} does not exist")
  endif()
  get_target_property(BUILTIN_LIB_SOURCES "${BUILTIN_LIB_TARGET_NAME}" SOURCES)
  list(LENGTH BUILTIN_LIB_SOURCES BUILTIN_LIB_SOURCES_LENGTH)
  if (BUILTIN_LIB_SOURCES_LENGTH EQUAL 0)
    message(FATAL_ERROR "Failed to find source files for ${arch} builtin library")
  endif()
  set(BUILTINS_LIT_SOURCE_FEATURES "")
  foreach (file_name ${BUILTIN_LIB_SOURCES})
    # Strip off any directories and file extensions. This approach means we add
    # add a single feature if there is a C source file or assembly override
    # present in the builtin library.
    # E.g.
    # "hexagon/udivsi3.S" => "udivsi3"
    # "udivsi3.c" => "udivsi3"
    get_filename_component(FILE_NAME_FILTERED "${file_name}" NAME_WE)
    list(APPEND BUILTINS_LIT_SOURCE_FEATURES "librt_has_${FILE_NAME_FILTERED}")
  endforeach()

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py
    )
  list(APPEND BUILTINS_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
endforeach()

# TODO: Add support for running tests on iOS and iOS simulator.

add_lit_testsuite(check-builtins "Running the Builtins tests"
  ${BUILTINS_TESTSUITES}
  DEPENDS ${BUILTINS_TEST_DEPS})
set_target_properties(check-builtins PROPERTIES FOLDER "Compiler-RT Misc")
