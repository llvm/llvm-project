// RUN: %clang -fsyntax-only -### %s 2>&1 | FileCheck --check-prefix=CHECK-DEF %s
// RUN: %clang -fsyntax-only -frelaxed-template-template-args %s 2>&1 | FileCheck --check-prefix=CHECK-ON %s
// RUN: %clang -fsyntax-only -fno-relaxed-template-template-args %s 2>&1 | FileCheck --check-prefix=CHECK-OFF %s
// RUN: %clang -fsyntax-only -fno-relaxed-template-template-args -Wno-deprecated-no-relaxed-template-template-args %s 2>&1 | FileCheck --check-prefix=CHECK-DIS --allow-empty %s

// CHECK-DEF-NOT: "-cc1"{{.*}} "-fno-relaxed-template-template-args"
// CHECK-ON:  warning: argument '-frelaxed-template-template-args' is deprecated [-Wdeprecated]
// CHECK-OFF: warning: argument '-fno-relaxed-template-template-args' is deprecated [-Wdeprecated-no-relaxed-template-template-args]
// CHECK-DIS-NOT: warning: argument '-fno-relaxed-template-template-args' is deprecated
