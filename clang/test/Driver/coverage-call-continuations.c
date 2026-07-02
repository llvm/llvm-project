// RUN: %clang -### -fprofile-instr-generate -fcoverage-mapping -fcoverage-call-continuations -c %s 2>&1 | FileCheck %s
// RUN: not %clang -### -fcoverage-call-continuations -c %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK: "-fcoverage-mapping"
// CHECK: "-fcoverage-call-continuations"

// ERR: error: invalid argument '-fcoverage-call-continuations' only allowed with '-fcoverage-mapping'
