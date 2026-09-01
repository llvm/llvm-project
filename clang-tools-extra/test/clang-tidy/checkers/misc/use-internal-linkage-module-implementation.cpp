// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang -std=c++20 --precompile %t/foo.cppm -o %t/foo.pcm
// RUN: %check_clang_tidy -std=c++20 %t/foo.cppm misc-use-internal-linkage %t/out
// RUN: %check_clang_tidy -std=c++20 %t/foo.cpp misc-use-internal-linkage %t/out \
// RUN:     -- -- -fmodule-file=foo=%t/foo.pcm

//--- foo.cppm

export module foo;

export void exported_fn();
export extern int exported_var;
export struct ExportedStruct;

//--- foo.cpp
module foo;

void exported_fn() {}
int exported_var;
struct ExportedStruct {};

void internal_fn() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'internal_fn' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static void internal_fn() {}
int internal_var;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'internal_var' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static int internal_var;
struct InternalStruct {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: struct 'InternalStruct' can be moved into an anonymous namespace to enforce internal linkage
