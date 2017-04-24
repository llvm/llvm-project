// RUN: %clang_cc1 -std=c11 -E %s | FileCheck %s
// RUN: %clang_cc1 -std=c11 -fms-compatibility -E %s | FileCheck %s
#include <stdatomic.h>

int bool_lock_free = ATOMIC_BOOL_LOCK_FREE;
// CHECK: bool_lock_free = {{ *[012] *;}}

int char_lock_free = ATOMIC_CHAR_LOCK_FREE;
// CHECK: char_lock_free = {{ *[012] *;}}

int char16_t_lock_free = ATOMIC_CHAR16_T_LOCK_FREE;
// CHECK: char16_t_lock_free = {{ *[012] *;}}

int char32_t_lock_free = ATOMIC_CHAR32_T_LOCK_FREE;
// CHECK: char32_t_lock_free = {{ *[012] *;}}

int wchar_t_lock_free = ATOMIC_WCHAR_T_LOCK_FREE;
// CHECK: wchar_t_lock_free = {{ *[012] *;}}

int short_lock_free = ATOMIC_SHORT_LOCK_FREE;
// CHECK: short_lock_free = {{ *[012] *;}}

int int_lock_free = ATOMIC_INT_LOCK_FREE;
// CHECK: int_lock_free = {{ *[012] *;}}

int long_lock_free = ATOMIC_LONG_LOCK_FREE;
// CHECK: long_lock_free = {{ *[012] *;}}

int llong_lock_free = ATOMIC_LLONG_LOCK_FREE;
// CHECK: llong_lock_free = {{ *[012] *;}}

int pointer_lock_free = ATOMIC_POINTER_LOCK_FREE;
// CHECK: pointer_lock_free = {{ *[012] *;}}
