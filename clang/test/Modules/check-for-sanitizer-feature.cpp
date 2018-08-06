// RUN: rm -rf %t.1 %t.2
// RUN: mkdir %t.1 %t.2

// Build and use an ASan-enabled module.
// RUN: %clang_cc1 -fsanitize=address -fmodules -fmodules-cache-path=%t.1 \
// RUN:   -fmodule-map-file=%S/Inputs/check-for-sanitizer-feature/map \
// RUN:   -I %S/Inputs/check-for-sanitizer-feature -verify %s
// RUN: ls %t.1 | count 2

// Force a module rebuild by disabling ASan.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t.1 \
// RUN:   -fmodule-map-file=%S/Inputs/check-for-sanitizer-feature/map \
// RUN:   -I %S/Inputs/check-for-sanitizer-feature -verify %s
// RUN: ls %t.1 | count 3

// Enable ASan again: check that there is no import failure, and no rebuild.
// RUN: %clang_cc1 -fsanitize=address -fmodules -fmodules-cache-path=%t.1 \
// RUN:   -fmodule-map-file=%S/Inputs/check-for-sanitizer-feature/map \
// RUN:   -I %S/Inputs/check-for-sanitizer-feature -verify %s
// RUN: ls %t.1 | count 3

// Some sanitizers can not affect AST generation when enabled. Check that
// module rebuilds don't occur when these sanitizers are enabled.
//
// First, build without any sanitization.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t.2 \
// RUN:   -fmodule-map-file=%S/Inputs/check-for-sanitizer-feature/map \
// RUN:   -I %S/Inputs/check-for-sanitizer-feature -verify %s
// RUN: ls %t.2 | count 2
//
// Next, build with sanitization, and check that a new module isn't built.
// RUN: %clang_cc1 -fsanitize=cfi-vcall,unsigned-integer-overflow,nullability-arg,null -fmodules \
// RUN:   -fmodules-cache-path=%t.2 \
// RUN:   -fmodule-map-file=%S/Inputs/check-for-sanitizer-feature/map \
// RUN:   -I %S/Inputs/check-for-sanitizer-feature -verify %s
// RUN: ls %t.2 | count 2

// Finally, test that including enabled sanitizers in the module hash isn't
// required to ensure correctness of module imports.
//
// Emit a PCH with ASan enabled.
// RUN: %clang_cc1 -x c -fsanitize=address %S/Inputs/check-for-sanitizer-feature/check.h -emit-pch -o %t.asan_pch
//
// Import the PCH without ASan enabled (we expect an error).
// RUN: not %clang_cc1 -x c -include-pch %t.asan_pch %s -verify 2>&1 | FileCheck %s --check-prefix=PCH_MISMATCH
// PCH_MISMATCH: AST file was compiled with the target feature '-fsanitize=address' but the current translation unit is not
//
// Emit a PCH with UBSan enabled.
// RUN: %clang_cc1 -x c -fsanitize=null %S/Inputs/check-for-sanitizer-feature/check.h -emit-pch -o %t.ubsan_pch
//
// Import the PCH without UBSan enabled (should work just fine).
// RUN: %clang_cc1 -x c -include-pch %t.ubsan_pch %s -I %S/Inputs/check-for-sanitizer-feature -verify

#include "check.h"

#if __has_feature(address_sanitizer)
#if HAS_ASAN != 1
#error Module doesn't have the address_sanitizer feature, but main program does.
#endif
#else
#if HAS_ASAN != 0
#error Module has the address_sanitizer feature, but main program doesn't.
#endif
#endif

// expected-no-diagnostics
