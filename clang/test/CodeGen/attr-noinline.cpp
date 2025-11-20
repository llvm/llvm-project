// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

bool bar();
void f(bool, bool);
void g(bool);

static int baz(int x) {
    return x * 10;
}

[[clang::noinline]] bool noi() { return true; }
[[msvc::noinline]] bool ms_noi() { return true; }

void foo(int i) {
  [[clang::noinline]] bar();
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR:[0-9]+]]
  [[clang::noinline]] i = baz(i);
// CHECK: call noundef i32 @_ZL3bazi({{.*}}) #[[NOINLINEATTR]]
  [[clang::noinline]] (i = 4, bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
  [[clang::noinline]] (void)(bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
  [[clang::noinline]] f(bar(), bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call void @_Z1fbb({{.*}}) #[[NOINLINEATTR]]
  [[clang::noinline]] [] { bar(); bar(); }(); // noinline only applies to the anonymous function call
// CHECK: call void @"_ZZ3fooiENK3$_0clEv"(ptr {{[^,]*}} %ref.tmp) #[[NOINLINEATTR]]
  [[clang::noinline]] for (bar(); bar(); bar()) {}
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
  bar();
// CHECK: call noundef zeroext i1 @_Z3barv()
  [[clang::noinline]] noi();
// CHECK: call noundef zeroext i1 @_Z3noiv()
  noi();
// CHECK: call noundef zeroext i1 @_Z3noiv()
  [[gnu::noinline]] bar();
// CHECK: call noundef zeroext i1 @_Z3barv()
}

void ms_noi_check(int i) {
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
// CHECK: call void @"_ZZ12ms_noi_checkiENK3$_0clEv"(ptr {{[^,]*}} %ref.tmp) #[[NOINLINEATTR]]
  [[msvc::noinline]] for (bar(); bar(); bar()) {}
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[NOINLINEATTR]]
  [[msvc::noinline]] ms_noi();
// CHECK: call noundef zeroext i1 @_Z6ms_noiv()
  ms_noi();
// CHECK: call noundef zeroext i1 @_Z6ms_noiv()
}

struct S {
  friend bool operator==(const S &LHS, const S &RHS);
};

void func(const S &s1, const S &s2) {
  [[clang::noinline]]g(s1 == s2);
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[NOINLINEATTR]]
// CHECK: call void @_Z1gb({{.*}}) #[[NOINLINEATTR]]
  bool b;
  [[clang::noinline]] b = s1 == s2;
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[NOINLINEATTR]]

  [[msvc::noinline]]g(s1 == s2);
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[NOINLINEATTR]]
// CHECK: call void @_Z1gb({{.*}}) #[[NOINLINEATTR]]
  [[msvc::noinline]] b = s1 == s2;
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[NOINLINEATTR]]
}

// CHECK: attributes #[[NOINLINEATTR]] = { noinline }
