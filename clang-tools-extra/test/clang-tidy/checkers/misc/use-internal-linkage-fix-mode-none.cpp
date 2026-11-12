// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- \
// RUN:   -config="{CheckOptions: {misc-use-internal-linkage.FixMode: 'None'}}"  -- -I%S/Inputs/use-internal-linkage

void func() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES-NOT: static void func() {}

int global;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'global' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES-NOT: static int global;
