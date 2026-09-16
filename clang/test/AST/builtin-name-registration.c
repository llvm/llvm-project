// Verify builtins register under their full name when formed with a prefix.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ast-dump %s | FileCheck %s

void use(void) {
  (void)__builtin_abs(-1);
  __builtin_ia32_pause();
}

// CHECK:      FunctionDecl {{.*}} implicit used __builtin_abs 'int (int)'
// CHECK:      BuiltinAttr
// CHECK:      FunctionDecl {{.*}} implicit used __builtin_ia32_pause 'void (void)'
// CHECK:      BuiltinAttr
