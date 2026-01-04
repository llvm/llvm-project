set(TSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(TSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(${COMPILER_RT_DEFAULT_TARGET_ARCH} MATCHES "(x86_64|s390x)")
  list(APPEND TSAN_TEST_DEPS GotsanRuntimeCheck)
endif()
list(APPEND TSAN_TEST_DEPS tsan)
if(COMPILER_RT_LIBCXX_PATH AND
   COMPILER_RT_LIBCXXABI_PATH AND
   COMPILER_RT_TEST_COMPILER_ID STREQUAL "Clang"
   AND NOT APPLE AND NOT ANDROID)
  list(APPEND TSAN_TEST_DEPS libcxx_tsan)
  set(TSAN_HAS_LIBCXX True)
else()
  set(TSAN_HAS_LIBCXX False)
endif()

set(TSAN_DYNAMIC_TEST_DEPS ${TSAN_TEST_DEPS})
set(TSAN_TESTSUITES)
set(TSAN_DYNAMIC_TESTSUITES)
set(TSAN_ENABLE_DOMINANCE_ANALYSIS "False") # Disable dominance analysis by default

if (NOT DEFINED TSAN_TEST_DEFLAKE_THRESHOLD)
  set(TSAN_TEST_DEFLAKE_THRESHOLD "10")
endif()

set(TSAN_TEST_ARCH ${TSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(TSAN_SUPPORTED_ARCH TSAN_TEST_ARCH)
endif()

# Unified function for generating TSAN test suites by architectures.
# Arguments:
#   OUT_LIST_VAR    - name of output list (for example, TSAN_TESTSUITES or TSAN_DOM_TESTSUITES)
#   SUFFIX_KIND     - string added to config suffix after "-${arch}" (for example, "" or "-dominance")
#   CONFIG_KIND     - string added to config name after "Config" (for example, "" or "Dominance")
#   ENABLE_DOM       - "True"/"False" enable dominance analysis
function(tsan_generate_arch_suites OUT_LIST_VAR SUFFIX_KIND CONFIG_KIND ENABLE_DOM)
  foreach(arch ${TSAN_TEST_ARCH})
    set(TSAN_ENABLE_DOMINANCE_ANALYSIS "${ENABLE_DOM}")

    set(TSAN_TEST_APPLE_PLATFORM "osx")
    set(TSAN_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_osx_MIN_VER_FLAG}")

    set(TSAN_TEST_TARGET_ARCH ${arch})
    string(TOLOWER "-${arch}${SUFFIX_KIND}" TSAN_TEST_CONFIG_SUFFIX)
    get_test_cc_for_arch(${arch} TSAN_TEST_TARGET_CC TSAN_TEST_TARGET_CFLAGS)

    string(REPLACE ";" " " LIBDISPATCH_CFLAGS_STRING " ${COMPILER_RT_TEST_LIBDISPATCH_CFLAGS}")
    string(APPEND TSAN_TEST_TARGET_CFLAGS ${LIBDISPATCH_CFLAGS_STRING})

    if (COMPILER_RT_HAS_MSSE4_2_FLAG)
      string(APPEND TSAN_TEST_TARGET_CFLAGS " -msse4.2 ")
    endif()

    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}Config${CONFIG_KIND})

    configure_lit_site_cfg(
            ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
            ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
            MAIN_CONFIG
            ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
    )
    list(APPEND ${OUT_LIST_VAR} ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})

    if(COMPILER_RT_TSAN_HAS_STATIC_RUNTIME)
      # Dynamic runtime for corresponding variant
      if("${SUFFIX_KIND}" STREQUAL "")
        string(TOLOWER "-${arch}-${OS_NAME}-dynamic" TSAN_TEST_CONFIG_SUFFIX)
        set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}DynamicConfig${CONFIG_KIND})
        list(APPEND TSAN_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
      else()
        string(TOLOWER "-${arch}-${OS_NAME}-dynamic${SUFFIX_KIND}" TSAN_TEST_CONFIG_SUFFIX)
        set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}DynamicConfig${CONFIG_KIND})
        # Track dynamic dominance-analysis suites separately for a dedicated target.
        list(APPEND TSAN_DOM_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
      endif()
      configure_lit_site_cfg(
              ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
              ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
              MAIN_CONFIG
              ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
      )
      list(APPEND ${OUT_LIST_VAR} ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
    endif()
  endforeach()

  # Propagate the assembled list to the parent scope
  set(${OUT_LIST_VAR} "${${OUT_LIST_VAR}}" PARENT_SCOPE)
  if(DEFINED TSAN_DOM_DYNAMIC_TESTSUITES)
    set(TSAN_DOM_DYNAMIC_TESTSUITES "${TSAN_DOM_DYNAMIC_TESTSUITES}" PARENT_SCOPE)
  endif()
endfunction()

# Default configuration
set(TSAN_TESTSUITES)
tsan_generate_arch_suites(TSAN_TESTSUITES "" "" "False")

# Enable dominance analysis (check-tsan-dominance-analysis target)
set(TSAN_DOM_TESTSUITES)
tsan_generate_arch_suites(TSAN_DOM_TESTSUITES "-dominance" "Dominance" "True")

# iOS and iOS simulator test suites
# These are not added into "check-all", in order to run these tests, use
# "check-tsan-iossim-x86_64" and similar. They also require an extra environment
# variable to select which iOS device or simulator to use, e.g.:
# SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER="iPhone 6"
if(APPLE)
  set(TSAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
  set(TSAN_APPLE_PLATFORMS ${TSAN_SUPPORTED_OS})
  foreach(platform ${TSAN_APPLE_PLATFORMS})
    if ("${platform}" STREQUAL "osx")
      # Skip macOS because it's handled by the code above that builds tests for the host machine.
      continue()
    endif()
    list_intersect(
      TSAN_TEST_${platform}_ARCHS
      TSAN_SUPPORTED_ARCH
      DARWIN_${platform}_ARCHS
    )
    foreach(arch ${TSAN_TEST_${platform}_ARCHS})
      get_test_cflags_for_apple_platform(
        "${platform}"
        "${arch}"
        TSAN_TEST_TARGET_CFLAGS
        )
      string(TOUPPER "${arch}" ARCH_UPPER_CASE)
      get_capitalized_apple_platform("${platform}" PLATFORM_CAPITALIZED)
      set(CONFIG_NAME "${PLATFORM_CAPITALIZED}${ARCH_UPPER_CASE}Config")
      set(TSAN_TEST_CONFIG_SUFFIX "-${arch}-${platform}")
      set(TSAN_TEST_APPLE_PLATFORM "${platform}")
      set(TSAN_TEST_TARGET_ARCH "${arch}")
      set(TSAN_TEST_MIN_DEPLOYMENT_TARGET_FLAG "${DARWIN_${platform}_MIN_VER_FLAG}")
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
        )
      add_lit_testsuite(check-tsan-${platform}-${arch} "ThreadSanitizer ${platform} ${arch} tests"
        ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/
        EXCLUDE_FROM_CHECK_ALL
        DEPENDS ${TSAN_TEST_DEPS})
    endforeach()
  endforeach()
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py)
if(COMPILER_RT_TSAN_HAS_STATIC_RUNTIME)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic/lit.site.cfg.py)
endif()
list(APPEND TSAN_TEST_DEPS TsanUnitTests)
list(APPEND TSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
if(COMPILER_RT_TSAN_HAS_STATIC_RUNTIME)
  list(APPEND TSAN_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic)
endif()
list(APPEND TSAN_DOM_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
if(COMPILER_RT_TSAN_HAS_STATIC_RUNTIME)
  list(APPEND TSAN_DOM_DYNAMIC_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/dynamic)
endif()

add_lit_testsuite(check-tsan "Running ThreadSanitizer tests"
  ${TSAN_TESTSUITES}
  DEPENDS ${TSAN_TEST_DEPS})
set_target_properties(check-tsan PROPERTIES FOLDER "Compiler-RT Tests")

if(COMPILER_RT_TSAN_HAS_STATIC_RUNTIME)
  add_lit_testsuite(check-tsan-dynamic "Running the ThreadSanitizer tests with dynamic runtime"
                    ${TSAN_DYNAMIC_TESTSUITES}
                    EXCLUDE_FROM_CHECK_ALL
                    DEPENDS ${TSAN_DYNAMIC_TEST_DEPS})
endif()

add_lit_testsuite(check-tsan-dominance-analysis "Running ThreadSanitizer tests (dominance analysis)"
        ${TSAN_DOM_TESTSUITES}
        DEPENDS ${TSAN_TEST_DEPS})
set_target_properties(check-tsan-dominance-analysis PROPERTIES FOLDER "Compiler-RT Tests")

# New target: dynamic + dominance analysis
if(COMPILER_RT_TSAN_HAS_STATIC_RUNTIME)
  add_lit_testsuite(check-tsan-dominance-analysis-dynamic "Running ThreadSanitizer tests (dynamic, dominance analysis)"
          ${TSAN_DOM_DYNAMIC_TESTSUITES}
          EXCLUDE_FROM_CHECK_ALL
          DEPENDS ${TSAN_DYNAMIC_TEST_DEPS})
  set_target_properties(check-tsan-dominance-analysis-dynamic PROPERTIES FOLDER "Compiler-RT Tests")
endif()
