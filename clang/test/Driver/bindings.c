// Basic binding.
// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -no-integrated-as %s 2>&1 | FileCheck %s --check-prefix=CHECK01
// CHECK01: "clang", inputs: ["{{.*}}bindings.c"], output: "{{.*}}.s"
// CHECK01: "GNU::Assembler", inputs: ["{{.*}}.s"], output: "{{.*}}.o"
// CHECK01: "gcc::Linker", inputs: ["{{.*}}.o"], output: "a.out"

// Clang control options

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK05
// CHECK05: "clang", inputs: ["{{.*}}bindings.c"], output: (nothing)

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -fsyntax-only -x c++ %s 2>&1 | FileCheck %s --check-prefix=CHECK08
// CHECK08: "clang", inputs: ["{{.*}}bindings.c"], output: (nothing)

// RUN: %clang -target i386-apple-darwin9 -ccc-print-bindings %s -S -arch ppc 2>&1 | FileCheck %s --check-prefix=CHECK11
// CHECK11: "clang", inputs: ["{{.*}}bindings.c"], output: "bindings.s"

// RUN: %clang -target powerpc-unknown-unknown -ccc-print-bindings %s -S 2>&1 | FileCheck %s --check-prefix=CHECK12
// CHECK12: "clang", inputs: ["{{.*}}bindings.c"], output: "bindings.s"

// Darwin bindings
// RUN: %clang -target i386-apple-darwin9 -no-integrated-as -ccc-print-bindings %s 2>&1 | FileCheck %s --check-prefix=CHECK14
// CHECK14: "clang", inputs: ["{{.*}}bindings.c"], output: "{{.*}}.s"
// CHECK14: "darwin::Assembler", inputs: ["{{.*}}.s"], output: "{{.*}}.o"
// CHECK14: "darwin::Linker", inputs: ["{{.*}}.o"], output: "a.out"

// GNU StaticLibTool binding
// RUN: %clang -target x86_64-linux-gnu -ccc-print-bindings --emit-static-lib %s 2>&1 | FileCheck %s --check-prefix=CHECK15
// CHECK15: "x86_64-unknown-linux-gnu" - "GNU::StaticLibTool", inputs: ["{{.*}}.o"], output: "a.out"

// Darwin StaticLibTool binding
// RUN: %clang -target i386-apple-darwin9 -ccc-print-bindings --emit-static-lib %s 2>&1 | FileCheck %s --check-prefix=CHECK16
// CHECK16: "i386-apple-darwin9" - "darwin::StaticLibTool", inputs: ["{{.*}}.o"], output: "a.out"
