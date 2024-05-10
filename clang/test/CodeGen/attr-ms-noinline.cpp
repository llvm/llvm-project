// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

bool bar();
void f(bool, bool);
void g(bool);

static int baz(int x) {
    return x * 10;
}

[[msvc::noinline]] bool noi() { }

void foo(int i) {
  [[msvc::noinline]] bar();
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR:[0-9]+]]
  [[msvc::noinline]] i = baz(i);
// CHECK: call noundef i32 @_ZL3bazi({{.*}}) #[[NOINLINEATTR]]
  [[msvc::noinline]] (i = 4, bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
  [[msvc::noinline]] (void)(bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
  [[msvc::noinline]] f(bar(), bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call void @_Z1fbb({{.*}}) #[[NOINLINEATTR]]
  [[msvc::noinline]] [] { bar(); bar(); }(); // noinline only applies to the anonymous function call
// CHECK: call void @"_ZZ3fooiENK3$_0clEv"(ptr {{[^,]*}} %ref.tmp) #[[NOINLINEATTR]]
  [[msvc::noinline]] for (bar(); bar(); bar()) {}
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
  bar();
// CHECK: call noundef zeroext i1 @_Z3barv()
  [[msvc::noinline]] noi();
// CHECK: call noundef zeroext i1 @_Z3noiv()
  noi();
// CHECK: call noundef zeroext i1 @_Z3noiv()
}

struct S {
  friend bool operator==(const S &LHS, const S &RHS);
};

void func(const S &s1, const S &s2) {
  [[msvc::noinline]]g(s1 == s2);
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[NOINLINEATTR]]
// CHECK: call void @_Z1gb({{.*}}) #[[NOINLINEATTR]]
  bool b;
  [[msvc::noinline]] b = s1 == s2;
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[NOINLINEATTR]]
}

// CHECK: attributes #[[NOINLINEATTR]] = { noinline }
