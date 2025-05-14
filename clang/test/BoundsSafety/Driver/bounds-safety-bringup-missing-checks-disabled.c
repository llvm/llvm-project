// =============================================================================
// All checks on
// =============================================================================

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --allow-empty --check-prefix=EMPTY %s

// =============================================================================
// batch_0 on
// =============================================================================

// RUN: %clang -fbounds-safety -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefix=EMPTY --allow-empty %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=EMPTY --allow-empty %s

// EMPTY-NOT: warning:

// =============================================================================
// All checks off
// =============================================================================

// RUN: %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=all \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=NONE %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=NONE %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=NONE %s

// NONE: warning: compiling with legacy -fbounds-safety bounds checks is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to use the new bound checks

// =============================================================================
// all on then one check off
// =============================================================================

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=AS-OFF-REM-FLAG %s
// AS-OFF-REM-FLAG: warning: compiling with "access_size" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=access_size to enable the new bound checks

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=indirect_count_update \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=ICU-OFF-REM-FLAG %s
// ICU-OFF-REM-FLAG: warning: compiling with "indirect_count_update" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=indirect_count_update to enable the new bound checks

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=return_size \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=RS-OFF-REM-FLAG %s
// RS-OFF-REM-FLAG: warning: compiling with "return_size" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=return_size to enable the new bound checks

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=ended_by_lower_bound \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=EBLB-OFF-REM-FLAG %s
// EBLB-OFF-REM-FLAG: warning: compiling with "ended_by_lower_bound" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=ended_by_lower_bound to enable the new bound checks

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=compound_literal_init \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CLI-OFF-REM-FLAG %s
// CLI-OFF-REM-FLAG: warning: compiling with "compound_literal_init" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=compound_literal_init to enable the new bound checks

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=LIBCA-OFF-REM-FLAG %s
// LIBCA-OFF-REM-FLAG: compiling with "libc_attributes" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=libc_attributes to enable the new bound checks

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=array_subscript_agg \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=ASA-OFF-REM-FLAG %s
// ASA-OFF-REM-FLAG: warning: compiling with "array_subscript_agg" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=array_subscript_agg to enable the new bound checks


// =============================================================================
// Individual checks enabled
// =============================================================================

// RUN: %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=all \
// RUN:   -fbounds-safety-bringup-missing-checks=access_size \
// RUN:   -fbounds-safety-bringup-missing-checks=indirect_count_update \
// RUN:   -fbounds-safety-bringup-missing-checks=return_size \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck \
// RUN:   --check-prefixes=EBLB-OFF-ADD-FLAG,CLI-OFF-ADD-FLAG,LIBCA-OFF-ADD-FLAG,ASA-OFF-ADD-FLAG %s

// RUN: %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=all \
// RUN:   -fbounds-safety-bringup-missing-checks=ended_by_lower_bound\
// RUN:   -fbounds-safety-bringup-missing-checks=compound_literal_init \
// RUN:   -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:   -fbounds-safety-bringup-missing-checks=array_subscript_agg \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck \
// RUN:   --check-prefixes=AS-OFF-ADD-FLAG,ICU-OFF-ADD-FLAG,RS-OFF-ADD-FLAG %s

// AS-OFF-ADD-FLAG: warning: compiling with "access_size" bounds check disabled is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to enable the new bound checks
// ICU-OFF-ADD-FLAG: warning: compiling with "indirect_count_update" bounds check disabled is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to enable the new bound checks
// RS-OFF-ADD-FLAG: warning: compiling with "return_size" bounds check disabled is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to enable the new bound checks
// EBLB-OFF-ADD-FLAG: warning: compiling with "ended_by_lower_bound" bounds check disabled is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to enable the new bound checks
// CLI-OFF-ADD-FLAG: warning: compiling with "compound_literal_init" bounds check disabled is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to enable the new bound checks
// LIBCA-OFF-ADD-FLAG: warning: compiling with "libc_attributes" bounds check disabled is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to enable the new bound checks
// ASA-OFF-ADD-FLAG: warning: compiling with "array_subscript_agg" bounds check disabled is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to enable the new bound checks


// =============================================================================
// all on then two checks off
// =============================================================================

// Don't be exhaustive to keep this test case smaller

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fno-bounds-safety-bringup-missing-checks=indirect_count_update \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=AS-OFF-REM-FLAG,ICU-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fno-bounds-safety-bringup-missing-checks=return_size \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=AS-OFF-REM-FLAG,RS-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fno-bounds-safety-bringup-missing-checks=ended_by_lower_bound \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=AS-OFF-REM-FLAG,EBLB-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fno-bounds-safety-bringup-missing-checks=compound_literal_init \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=AS-OFF-REM-FLAG,CLI-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=AS-OFF-REM-FLAG,LIBCA-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fno-bounds-safety-bringup-missing-checks=array_subscript_agg \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=AS-OFF-REM-FLAG,ASA-OFF-REM-FLAG %s

// =============================================================================
// batch_0 on then one check off
// =============================================================================

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=AS-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=indirect_count_update \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=ICU-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=return_size \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=RS-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=ended_by_lower_bound \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=EBLB-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=compound_literal_init \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=CLI-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=LIBCA-OFF-REM-FLAG %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=array_subscript_agg \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=ASA-OFF-REM-FLAG %s

// =============================================================================
// batch_0 on then all off, then enable some
//
// In this case we shouldn't suggest removing `-fno-bounds-safety-bringup-missing-checks=<check>`
// because it was never provided by the user.
// =============================================================================

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=all \
// RUN:   -fbounds-safety-bringup-missing-checks=access_size \
// RUN:   -fbounds-safety-bringup-missing-checks=indirect_count_update \
// RUN:   -fbounds-safety-bringup-missing-checks=return_size \
// RUN:   -fbounds-safety-bringup-missing-checks=ended_by_lower_bound \
// RUN:   -fbounds-safety-bringup-missing-checks=compound_literal_init \
// RUN:   -fsyntax-only %s 2>&1 | \
// RUN: FileCheck --check-prefixes=LIBCA-OFF-ADD-FLAG,ASA-OFF-ADD-FLAG %s

// =============================================================================
// Check the diagnostics can be made into errors
// =============================================================================

// RUN: not %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=all \
// RUN:   -Werror=bounds-safety-legacy-checks-enabled %s 2>&1 | \
// RUN: FileCheck --check-prefix=NONE-ERROR %s

// RUN: not %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=all \
// RUN:   -Werror %s 2>&1 | \
// RUN: FileCheck --check-prefix=NONE-ERROR %s

// RUN: not %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -Werror %s 2>&1 | \
// RUN: FileCheck --check-prefix=NONE-ERROR %s

// NONE-ERROR: error: compiling with legacy -fbounds-safety bounds checks is deprecated; compile with -fbounds-safety-bringup-missing-checks=batch_0 to use the new bound checks

// RUN: not %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -Werror=bounds-safety-legacy-checks-enabled %s 2>&1 | \
// RUN: FileCheck --check-prefix=AS-OFF-REM-FLAG-ERROR %s

// AS-OFF-REM-FLAG-ERROR: error: compiling with "access_size" bounds check disabled is deprecated; remove -fno-bounds-safety-bringup-missing-checks=access_size to enable the new bound checks

// =============================================================================
// Check the diagnostics can be suppressed
// =============================================================================

// RUN: %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=all \
// RUN:   -fsyntax-only -Werror \
// RUN:   -Wno-bounds-safety-legacy-checks-enabled %s

// RUN: %clang -fbounds-safety \
// RUN:   -fno-bounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fsyntax-only -Werror \
// RUN:   -Wno-bounds-safety-legacy-checks-enabled %s

// RUN: %clang -fbounds-safety \
// RUN:   -fbounds-safety-bringup-missing-checks=batch_0 \
// RUN:   -fno-bounds-safety-bringup-missing-checks=access_size \
// RUN:   -fsyntax-only -Werror \
// RUN:   -Wno-bounds-safety-legacy-checks-enabled %s
