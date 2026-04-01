// RUN: %clang_cc1 -triple x86_64-windows-msvc -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN:   FileCheck %s

// Verify that compiler-generated atexit destructors for global variables have
// a non-empty linkageName in their DISubprogram. Without this, different
// template instantiations would be indistinguishable in the debug info.

struct Foo {
  ~Foo();
};

template <typename T>
struct Bar {
  static Foo &get() {
    static Foo instance;
    return instance;
  }
};

void use() {
  Bar<int>::get();
  Bar<float>::get();
}

// Both atexit destructors should have distinct linkageNames.
// CHECK-DAG: distinct !DISubprogram({{.*}}linkageName: "??__Finstance@?2??get@?$Bar@H@@SAAEAUFoo@@XZ@YAXXZ"
// CHECK-DAG: distinct !DISubprogram({{.*}}linkageName: "??__Finstance@?2??get@?$Bar@M@@SAAEAUFoo@@XZ@YAXXZ"
