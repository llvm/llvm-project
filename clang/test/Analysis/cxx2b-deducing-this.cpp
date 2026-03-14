// RUN: %clang_analyze_cc1 -std=c++2b -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection

template <typename T> void clang_analyzer_dump(T);

struct S {
  int num;
  S *orig;

  void a(this auto Self) {
    clang_analyzer_dump(&Self);     // expected-warning {{&Self}}
    clang_analyzer_dump(Self.orig); // expected-warning {{&s}}
    clang_analyzer_dump(Self.num);       // expected-warning {{5 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{5 S32b}}

    Self.num = 1;
    clang_analyzer_dump(Self.num);       // expected-warning {{1 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{5 S32b}}
  }

  void b(this auto& Self) {
    clang_analyzer_dump(&Self);     // expected-warning {{&s}}
    clang_analyzer_dump(Self.orig); // expected-warning {{&s}}
    clang_analyzer_dump(Self.num);       // expected-warning {{5 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{5 S32b}}

    Self.num = 2;
    clang_analyzer_dump(Self.num);       // expected-warning {{2 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{2 S32b}}
  }

  void c(this S Self) {
    clang_analyzer_dump(&Self);     // expected-warning {{&Self}}
    clang_analyzer_dump(Self.orig); // expected-warning {{&s}}
    clang_analyzer_dump(Self.num);       // expected-warning {{2 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{2 S32b}}

    Self.num = 3;
    clang_analyzer_dump(Self.num);       // expected-warning {{3 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{2 S32b}}
  }

  void c(this S Self, int I) {
    clang_analyzer_dump(I); // expected-warning {{11 S32b}}
    clang_analyzer_dump(&Self);     // expected-warning {{&Self}}
    clang_analyzer_dump(Self.orig); // expected-warning {{&s}}
    clang_analyzer_dump(Self.num);       // expected-warning {{2 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{2 S32b}}

    Self.num = 4;
    clang_analyzer_dump(Self.num);       // expected-warning {{4 S32b}}
    clang_analyzer_dump(Self.orig->num); // expected-warning {{2 S32b}}
  }
};

void top() {
  S s = {/*num=*/5, /*orig=*/&s};
  s.a();
  s.b(); // This call changes 's.num' to 2.
  s.c();
  s.c(11);
}


struct S2 {
  bool operator==(this auto, S2) {
    return true;
  }
};
void use_deducing_this() {
  int result = S2{} == S2{}; // no-crash
  clang_analyzer_dump(result); // expected-warning {{1 S32b}}
}
