// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=vptr -fsanitize-trap=vptr -emit-llvm %s -o - | FileCheck %s

struct A {
  virtual void foo();
};
struct B {
  virtual void bar();
};

void A::foo() { }
void B::bar() { }

int dynamic_type_cache_miss() {
  B b;
  A &a = reinterpret_cast<A&>(b);
  a.foo();
  return 0;
}

// CHECK: call void @llvm.ubsantrap(i8 4) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer
