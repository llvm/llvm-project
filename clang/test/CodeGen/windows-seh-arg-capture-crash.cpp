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

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@f@@"(ptr noundef %exception_pointers
// CHECK: entry:
// CHECK:   %frame_pointer.addr = alloca ptr, align 8
// CHECK:   %exception_pointers.addr = alloca ptr, align 8
// CHECK:   %0 = call ptr @llvm.eh.recoverfp(ptr @"?f@@YAXVspan_a@@@Z", ptr %frame_pointer)
// CHECK:   %input = call ptr @llvm.localrecover(ptr @"?f@@YAXVspan_a@@@Z", ptr %0, i32 0)


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

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@f@@.1"(ptr noundef %exception_pointers
// CHECK: entry:
// CHECK:   %frame_pointer.addr = alloca ptr, align 8
// CHECK:   %exception_pointers.addr = alloca ptr, align 8
// CHECK:   %0 = call ptr @llvm.eh.recoverfp(ptr @"?f@@YAXVspan_b@@@Z", ptr %frame_pointer)
// CHECK:   %1 = call ptr @llvm.localrecover(ptr @"?f@@YAXVspan_b@@@Z", ptr %0, i32 0)
// CHECK:   %input = load ptr, ptr %1, align 8
