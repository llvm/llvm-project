set(DASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Checking globals requires HSA Loader API 1.04 to be availabe.
include(CheckCSourceRuns)
include(CMakePushCheckState)
cmake_push_check_state()
set(CMAKE_REQUIRED_LIBRARIES hsa-runtime64)
check_c_source_runs([[
#include <stdbool.h>
#include <stdint.h>
typedef uint32_t hsa_status_t;
hsa_status_t hsa_init(void);
hsa_status_t hsa_shut_down(void);
hsa_status_t hsa_system_extension_supported(uint16_t, uint16_t, uint16_t,
                                            bool *);
int main(void) {
  if (hsa_init())
    return 1;
  bool ok = false;
  hsa_status_t st = hsa_system_extension_supported(0x201, 1, 4, &ok);
  hsa_shut_down();
  return st == 0 && ok ? 0 : 1;
}
]] DASAN_GLOBAL_SUPPORT)
cmake_pop_check_state()
if(DASAN_GLOBAL_SUPPORT)
  set(DASAN_GLOBAL_SUPPORT True)
else()
  set(DASAN_GLOBAL_SUPPORT False)
endif()

set(DASAN_TESTSUITES)
set(DASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
# Only the host half. A device test also links the copy of this runtime built
# for the GPU, which another runtimes build produces and which nothing here can
# depend on, so the two are kept in step by building the runtimes together.
list(APPEND DASAN_TEST_DEPS dasan)
set(DASAN_TEST_ARCH ${DASAN_SUPPORTED_ARCH})

foreach(arch ${DASAN_TEST_ARCH})
  set(DASAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" DASAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} DASAN_TEST_TARGET_CC DASAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND DASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-dasan "Running the DeviceAddressSanitizer tests"
  ${DASAN_TESTSUITES}
  DEPENDS ${DASAN_TEST_DEPS})
set_target_properties(check-dasan PROPERTIES FOLDER "compiler-rt Misc")
