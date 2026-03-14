// RUN: %clang_cc1 -x c++ -include %s -emit-llvm-only %s
// RUN: %clang_cc1 -x c++ -emit-pch %s -o %t
// RUN: %clang_cc1 -include-pch %t -emit-llvm-only %s

#ifndef HEADER
#define HEADER

struct S00 { virtual void f(); };
struct S01 { virtual void f(); };
struct S02 { virtual void f(); };
struct S03 { virtual void f(); };
struct S04 { virtual void f(); };
struct S05 { virtual void f(); };
struct S06 { virtual void f(); };
struct S07 { virtual void f(); };
struct S08 { virtual void f(); };
struct S09 { virtual void f(); };
struct S10 { virtual void f(); };
struct S11 { virtual void f(); };
struct S12 { virtual void f(); };
struct S13 { virtual void f(); };
struct S14 { virtual void f(); };
struct S15 { virtual void f(); };
struct S16 { virtual void f(); };
struct S17 { virtual void f(); };
struct S18 { virtual void f(); };
struct S19 { virtual void f(); };
struct S20 { virtual void f(); };
struct S21 { virtual void f(); };
struct S22 { virtual void f(); };
struct S23 { virtual void f(); };
struct S24 { virtual void f(); };
struct S25 { virtual void f(); };
struct S26 { virtual void f(); };
struct S27 { virtual void f(); };
struct S28 { virtual void f(); };
struct S29 { virtual void f(); };
struct S30 { virtual void f(); };
struct S31 { virtual void f(); };
struct S32 { virtual void f(); };
struct S33 { virtual void f(); };
struct S34 { virtual void f(); };
struct S35 { virtual void f(); };
struct S36 { virtual void f(); };
struct S37 { virtual void f(); };
struct S38 { virtual void f(); };
struct S39 { virtual void f(); };
struct S40 { virtual void f(); };
struct S41 { virtual void f(); };
struct S42 { virtual void f(); };
struct S43 { virtual void f(); };
struct S44 { virtual void f(); };
struct S45 { virtual void f(); };
struct S46 { virtual void f(); };
struct S47 { virtual void f(); };
struct S48 { virtual void f(); };
struct S49 { virtual void f(); };
struct S50 { virtual void f(); };
struct S51 { virtual void f(); };
struct S52 { virtual void f(); };
struct S53 { virtual void f(); };
struct S54 { virtual void f(); };
struct S55 { virtual void f(); };
struct S56 { virtual void f(); };
struct S57 { virtual void f(); };
struct S58 { virtual void f(); };
struct S59 { virtual void f(); };
struct S60 { virtual void f(); };
struct S61 { virtual void f(); };
struct S62 { virtual void f(); };
struct S63 { virtual void f(); };
struct S64 { virtual void f(); };
struct S65 { virtual void f(); };
struct S66 { virtual void f(); };
struct S67 { virtual void f(); };
struct S68 { virtual void f(); };
struct S69 { virtual void f(); };

struct Test {
  // Deserializing this key function should cause the key functions
  // table to get resized.
  virtual void f(S00, S01, S02, S03, S04, S05, S06, S07, S08, S09,
                 S10, S11, S12, S13, S14, S15, S16, S17, S18, S19,
                 S20, S21, S22, S23, S24, S25, S26, S27, S28, S29,
                 S30, S31, S32, S33, S34, S35, S36, S37, S38, S39,
                 S40, S41, S42, S43, S44, S45, S46, S47, S48, S49,
                 S50, S51, S52, S53, S54, S55, S56, S57, S58, S59,
                 S60, S61, S62, S63, S64, S65, S66, S67, S68, S69);
  virtual void g();
};

#else

void Test::g() {}
void h(Test &t) { t.g(); }

#endif
