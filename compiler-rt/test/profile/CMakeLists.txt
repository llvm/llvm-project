set(PROFILE_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROFILE_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(PROFILE_TESTSUITES)
# Profile tests rely on the compiler-rt-headers being in the resource directory
set(PROFILE_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} compiler-rt-headers)
list(APPEND PROFILE_TEST_DEPS profile)
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND PROFILE_TEST_DEPS llvm-cov llvm-lto llvm-profdata opt)
  if(COMPILER_RT_HAS_LLD AND "lld" IN_LIST LLVM_ENABLE_PROJECTS)
    list(APPEND PROFILE_TEST_DEPS lld)
  endif()
endif()

set(PROFILE_TEST_ARCH ${PROFILE_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(PROFILE_SUPPORTED_ARCH PROFILE_TEST_ARCH)
endif()

pythonize_bool(LLVM_ENABLE_CURL)

foreach(arch ${PROFILE_TEST_ARCH})
  set(PROFILE_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} PROFILE_TEST_TARGET_CC PROFILE_TEST_TARGET_CFLAGS)
  # On MSVC, Profile-* tests must link with the same CRT model as the
  # clang_rt.profile static archive they exercise. When that archive pulls
  # in RTInterception / RTSanitizerCommon object libraries, those are built
  # with MultiThreadedDLL (/MD), so the .objs reference __imp_* symbols;
  # the test binary defaults to /MT and fails to link (LNK2019 __imp__stricmp
  # from interception_win.cpp, LNK4098 default-lib conflicts). Match the
  # DLL CRT here so test executables link against the same runtime.
  if(MSVC AND COMPILER_RT_HAS_INTERCEPTION AND NOT COMPILER_RT_PROFILE_BAREMETAL)
    string(APPEND PROFILE_TEST_TARGET_CFLAGS
           " -D_MT -D_DLL -Wl,-nodefaultlib:libcmt,-defaultlib:msvcrt,-defaultlib:oldnames")
  endif()
  set(CONFIG_NAME Profile-${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
   )
  list(APPEND PROFILE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-profile "Running the profile tests"
  ${PROFILE_TESTSUITES}
  DEPENDS ${PROFILE_TEST_DEPS})
