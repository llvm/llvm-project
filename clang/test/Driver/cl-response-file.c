// Test that we use the Windows tokenizer for clang-cl response files. The
// trailing backslash before the space should be interpreted as a literal
// backslash. PR23709



// RUN: printf '%%s\n' '/I%S\Inputs\cl-response-file\ /DFOO=2' > %t.rsp
// RUN: %clang_cl /c -### @%t.rsp -- %s 2>&1 | FileCheck %s

// CHECK: "-I" "{{.*}}\\Inputs\\cl-response-file\\" "-D" "FOO=2"
