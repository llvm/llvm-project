// DEFINE: %{base_cmd} = %clang_analyze_cc1 %s \
// DEFINE:   -analyzer-checker=security.insecureAPI.DeprecatedOrUnsafeBufferHandling

// DEFINE: %{verify_flag} =

// DEFINE: %{config_flag_unset} =
// DEFINE: %{config_flag_all} = -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=all
// DEFINE: %{config_flag_actionable} = -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=actionable
// DEFINE: %{config_flag_c11_only} = -analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=c11-only
// DEFINE: %{config_flag} = %{config_flag_unset}

// DEFINE: %{std_flag_c99} = -std=gnu99
// DEFINE: %{std_flag_c11} = -std=gnu11
// DEFINE: %{std_flag} = %{std_flag_c99}

// DEFINE: %{annexk_defines_unset} =
// DEFINE: %{annexk_defines_set} = -D__STDC_LIB_EXT1__=200509L -D__STDC_WANT_LIB_EXT1__=1
// DEFINE: %{annexk_flag} = %{annexk_defines_unset} 

// DEFINE: %{run_cmd} = %{base_cmd} %{verify_flag} %{std_flag} %{annexk_flag} %{config_flag}

// These cases should warn

// REDEFINE: %{verify_flag} = -verify=common
// REDEFINE: %{std_flag} = %{std_flag_c99}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_all}
// RUN: %{run_cmd}

// C99 with "all" mode
// REDEFINE: %{verify_flag} = -verify=common
// REDEFINE: %{std_flag} = %{std_flag_c99}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_all}
// RUN: %{run_cmd}

// C11 with default mode
// REDEFINE: %{verify_flag} = -verify=common
// REDEFINE: %{std_flag} = %{std_flag_c11}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_unset}
// RUN: %{run_cmd}

// C11 with "all" mode
// REDEFINE: %{verify_flag} = -verify=common
// REDEFINE: %{std_flag} = %{std_flag_c11}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_all}
// RUN: %{run_cmd}

// C11 with "c11-only" mode
// REDEFINE: %{verify_flag} = -verify=common
// REDEFINE: %{std_flag} = %{std_flag_c11}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_c11_only}
// RUN: %{run_cmd}

// C11 with "actionable" mode and Annex K available
// REDEFINE: %{verify_flag} = -verify=common
// REDEFINE: %{std_flag} = %{std_flag_c11}
// REDEFINE: %{annexk_flag} = %{annexk_defines_set}
// REDEFINE: %{config_flag} = %{config_flag_actionable}
// RUN: %{run_cmd}

// These cases should not warn

// C99 with default mode
// REDEFINE: %{verify_flag} = -verify=c99-default
// REDEFINE: %{std_flag} = %{std_flag_c99}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_unset}
// RUN: %{run_cmd}

// C99 with "actionable" mode and no Annex K
// REDEFINE: %{verify_flag} = -verify=c99-actionable
// REDEFINE: %{std_flag} = %{std_flag_c99}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_actionable}
// RUN: %{run_cmd}

// C99 with "c11-only" mode
// REDEFINE: %{verify_flag} = -verify=c99-c11only
// REDEFINE: %{std_flag} = %{std_flag_c99}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_c11_only}
// RUN: %{run_cmd}

// C11 with "actionable" mode and no Annex K
// REDEFINE: %{verify_flag} = -verify=c11-actionable-noannex
// REDEFINE: %{std_flag} = %{std_flag_c11}
// REDEFINE: %{annexk_flag} = %{annexk_defines_unset}
// REDEFINE: %{config_flag} = %{config_flag_actionable}
// RUN: %{run_cmd}


#include "Inputs/system-header-simulator.h"

extern char buf[128];
extern char src[128];

// c99-default-no-diagnostics
// c99-actionable-no-diagnostics
// c99-c11only-no-diagnostics
// c11-actionable-noannex-no-diagnostics

void test_memcpy(void) {
  memcpy(buf, src, 10);
  // common-warning@-1{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard}}
}

void test_memset(void) {
  memset(buf, 0, 10);
  // common-warning@-1{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard}}
}

void test_memmove(void) {
  memmove(buf, src, 10);
  // common-warning@-1{{Call to function 'memmove' is insecure as it does not provide security checks introduced in the C11 standard}}
}

