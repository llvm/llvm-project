// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions -fexceptions -emit-llvm -o - %s | FileCheck %s

class span_a {
 public:
  char data_;
  int size_;
};

long g(span_a input);

void f(span_a input) {
  __try {
  } __except (g(input)) {
  }
}

// CHECK-LABEL: define dso_local void @"?f@@YAXVspan_a@@@Z"(i64 %input.coerce)
// CHECK: entry:
// CHECK:   call void (...) @llvm.localescape(ptr %input)

typedef __SIZE_TYPE__ size_t;

class span_b {
 public:
  char data_;
  size_t size_;
};

long g(span_b input);

void f(span_b input) {
  __try {
  } __except (g(input)) {
  }
}

// CHECK-LABEL: define dso_local void @"?f@@YAXVspan_b@@@Z"(ptr noundef dead_on_return %input)
// CHECK: entry:
// CHECK:   %input.spill = alloca ptr, align 8
// CHECK:   call void (...) @llvm.localescape(ptr %input.spill)
