// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang -fms-extensions -rewrite-objc %t.mm -o %t-rw.cpp
// RUN: FileCheck -check-prefix CHECK-LP --input-file=%t-rw.cpp %s
// RUN: %clang -g -fms-extensions -rewrite-objc %t.mm -o %t-rw.cpp
// RUN: FileCheck -check-prefix CHECK-LPG --input-file=%t-rw.cpp %s
// rdar://13138170

int z();

int x() {
    id foo;
    for (id y in foo) {
        z();
    }
    return 0;
}
// CHECK-LP-NOT: #line
// CHECK-LPG: #line
