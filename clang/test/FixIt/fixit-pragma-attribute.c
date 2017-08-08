// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// Verify that the suggested attribute subject match rules don't include the
// rules that are not applicable in the current language mode.

#pragma clang attribute push (__attribute__((abi_tag("a"))))
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:60-[[@LINE-1]]:60}:", apply_to = any(record(unless(is_union)), variable, function)"
