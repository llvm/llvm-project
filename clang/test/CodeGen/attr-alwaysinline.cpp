// RUN: %clang_cc1 -S -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

bool bar();
void f(bool, bool);
void g(bool);

void foo(int i) {
  [[clang::always_inline]] bar();
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR:[0-9]+]]
  [[clang::always_inline]] (i = 4, bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR]]
  [[clang::always_inline]] (void)(bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR]]
  [[clang::always_inline]] f(bar(), bar());
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR]]
// CHECK: call void @_Z1fbb({{.*}}) #[[ALWAYSINLINEATTR]]
  [[clang::always_inline]] for (bar(); bar(); bar()) {}
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ALWAYSINLINEATTR]]
  bar();
// CHECK: call noundef zeroext i1 @_Z3barv()
  [[gnu::always_inline]] bar();
// CHECK: call noundef zeroext i1 @_Z3barv()
}

struct S {
  friend bool operator==(const S &LHS, const S &RHS);
};

void func(const S &s1, const S &s2) {
  [[clang::always_inline]]g(s1 == s2);
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[ALWAYSINLINEATTR]]
// CHECK: call void @_Z1gb({{.*}}) #[[ALWAYSINLINEATTR]]
  bool b;
  [[clang::always_inline]] b = s1 == s2;
// CHECK: call noundef zeroext i1 @_ZeqRK1SS1_({{.*}}) #[[ALWAYSINLINEATTR]]
}

// CHECK: attributes #[[ALWAYSINLINEATTR]] = { alwaysinline }
