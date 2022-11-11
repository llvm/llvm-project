// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-output=text -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-output=text -std=c++17 -verify %s

#include "Inputs/system-header-simulator-cxx.h"

namespace copyMoveTrackCtor {
struct S {
  int *p1, *p2;
  S(int *a, int *b) : p1(a), p2(b) {} // expected-note{{Null pointer value stored to 's.p1'}}
};

void CtorDirect() {
  int *x = nullptr, *y = nullptr; 
  // expected-note@-1{{'x' initialized to a null pointer value}}

  S s(x, y); 
  // expected-note@-1{{Passing null pointer value via 1st parameter 'a'}}
  // expected-note@-2{{Calling constructor for 'S'}}
  // expected-note@-3{{Returning from constructor for 'S'}}
  // expected-note@-4{{'s' initialized here}}
  S s2 = s; // expected-note{{Null pointer value stored to 's2.p1'}}
  // expected-note@-1{{'s2' initialized here}}
  S s3 = s2;  // expected-note{{Null pointer value stored to 's3.p1'}}
  // expected-note@-1{{'s3' initialized here}}
  S s4 = std::move(s3); // expected-note{{Null pointer value stored to 's4.p1'}}
  // expected-note@-1{{'s4' initialized here}}
  S s5 = s4; // expected-note{{Null pointer value stored to 's5.p1'}}

  int i = *s5.p1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1{{Dereference of null pointer (loaded from field 'p1')}}

  (void) i;
}
} // namespace copyMoveTrackCtor

namespace copyMoveTrackInitList {
struct S {
  int *p1, *p2;
};

void InitListDirect() {
  int *x = nullptr, *y = nullptr; //expected-note{{'x' initialized to a null pointer value}}

  S s{x, y}; //expected-note{{'s.p1' initialized to a null pointer value}}
  //expected-note@-1{{'s' initialized here}}
  S s2 = s; // expected-note{{Null pointer value stored to 's2.p1'}}
  // expected-note@-1{{'s2' initialized here}}
  S s3 = s2; // expected-note{{Null pointer value stored to 's3.p1'}}
  // expected-note@-1{{'s3' initialized here}}
  S s4 = std::move(s3); // expected-note{{Null pointer value stored to 's4.p1'}}
  // expected-note@-1{{'s4' initialized here}}
  S s5 = s4; // expected-note{{Null pointer value stored to 's5.p1'}}

  int i = *s5.p1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1{{Dereference of null pointer (loaded from field 'p1')}}

  (void) i;
}

void InitListAssign() {
  int *x = nullptr, *y = nullptr; //expected-note{{'x' initialized to a null pointer value}}

  S s = {x, y}; //expected-note{{'s.p1' initialized to a null pointer value}}
  //expected-note@-1{{'s' initialized here}}
  S s2 = s; // expected-note{{Null pointer value stored to 's2.p1'}}
  // expected-note@-1{{'s2' initialized here}}
  S s3 = s2; // expected-note{{Null pointer value stored to 's3.p1'}}
  // expected-note@-1{{'s3' initialized here}}
  S s4 = std::move(s3); // expected-note{{Null pointer value stored to 's4.p1'}}
  // expected-note@-1{{'s4' initialized here}}
  S s5 = s4; // expected-note{{Null pointer value stored to 's5.p1'}}

  int i = *s5.p1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1{{Dereference of null pointer (loaded from field 'p1')}}

  (void) i;
}

} // namespace copyMoveTrackInitList

namespace copyMoveTrackCtorMemberInitList {
struct S {
  int *p1, *p2;
  S(int *a, int *b) : p1{a}, p2{b} {} // expected-note{{Null pointer value stored to 's.p1'}}
};

void CtorDirect() {
  int *x = nullptr, *y = nullptr; 
  // expected-note@-1{{'x' initialized to a null pointer value}}

  S s{x, y}; 
  // expected-note@-1{{Passing null pointer value via 1st parameter 'a'}}
  // expected-note@-2{{Calling constructor for 'S'}}
  // expected-note@-3{{Returning from constructor for 'S'}}
  // expected-note@-4{{'s' initialized here}}
  S s2 = s; // expected-note{{Null pointer value stored to 's2.p1'}}
  // expected-note@-1{{'s2' initialized here}}
  S s3 = s2;  // expected-note{{Null pointer value stored to 's3.p1'}}
  // expected-note@-1{{'s3' initialized here}}
  S s4 = std::move(s3); // expected-note{{Null pointer value stored to 's4.p1'}}
  // expected-note@-1{{'s4' initialized here}}
  S s5 = s4; // expected-note{{Null pointer value stored to 's5.p1'}}

  int i = *s5.p1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1{{Dereference of null pointer (loaded from field 'p1')}}

  (void) i;
}
} // namespace copyMoveTrackCtorMemberInitList

namespace directInitList {
struct S {
  int *p1, *p2;
};

void InitListDirect() {
  int *x = nullptr, *y = nullptr; //expected-note{{'y' initialized to a null pointer value}}

  S s{x, y}; //expected-note{{'s.p2' initialized to a null pointer value}}

  int i = *s.p2; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1{{Dereference of null pointer}}
  (void) i;
}
} // namespace directInitList

namespace directNestedInitList {
struct S2 {
  int *p1, *p2;
};

struct S {
  S2 s;
};

void InitListNestedDirect() {
  int *x = nullptr, *y = nullptr; //expected-note{{'y' initialized to a null pointer value}}

  //FIXME: Put more information to the notes.
  S s{x, y}; //expected-note{{'s.s.p2' initialized to a null pointer value}}

  int i = *s.s.p2; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1{{Dereference of null pointer}}
  (void) i;
}
} // namespace directNestedInitList

#if __cplusplus >= 201703L

namespace structuredBinding {
struct S {
  int *p1, *p2;
};

void StructuredBinding() {
  int *x = nullptr, *y = nullptr;
  //expected-note@-1{{'y' initialized to a null pointer value}}

  S s{x, y}; 
  //expected-note@-1{{'s.p2' initialized to a null pointer value}}
  //expected-note@-2{{'s' initialized here}}

  auto [a, b] = s; //expected-note{{Null pointer value stored to '.p2'}}

  int i = *b; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1{{Dereference of null pointer}}
  (void) i;
}
} // namespace structuredBinding

#endif

namespace nestedCtorInitializer {
  struct S5{
    int *x, *y;
  };

  struct S4 {
    S5 s5;
  };

  struct S3 {
    S4 s4;
  };

  struct S2 {
    S3 s3;
  };

  struct S {
    S2 s2;

    //FIXME: Put more information to the notes.
    S(int *x, int *y) : s2{x, y} {}; 
    // expected-note@-1{{Null pointer value stored to 's.s2.s3.s4.s5.y'}}
  };

  void nestedCtorInit(){
    int *x = nullptr, *y = nullptr; // expected-note{{'y' initialized to a null pointer value}}

    S s{x,y}; 
    // expected-note@-1{{Passing null pointer value via 2nd parameter}}
    // expected-note@-2{{Calling constructor for 'S'}}
    // expected-note@-3{{Returning from constructor for 'S'}}

    int i = *s.s2.s3.s4.s5.y; // expected-warning{{Dereference of null pointer}}
    // expected-note@-1{{Dereference of null pointer}}
    (void) i;
  }
} // namespace nestedCtorInitializer

namespace NestedRegionTrack {
struct N {
  int *e;
};

struct S {
  N y;
};

void NestedRegionTrack() {
  int *x = nullptr, *y = nullptr;
  // expected-note@-1{{'y' initialized to a null pointer value}}

  // Test for nested single element initializer list here.
  S a{{{{{{{{y}}}}}}}};
  // expected-note@-1{{'a.y.e' initialized to a null pointer value}}
  // expected-note@-2{{'a' initialized here}}
  // expected-warning@-3{{too many braces around scalar initializer}}
  // expected-warning@-4{{too many braces around scalar initializer}}
  // expected-warning@-5{{too many braces around scalar initializer}}
  // expected-warning@-6{{too many braces around scalar initializer}}
  // expected-warning@-7{{too many braces around scalar initializer}}

  S b = a; // expected-note{{Null pointer value stored to 'b.y.e'}}

  int i = *b.y.e;
  // expected-warning@-1{{Dereference of null pointer}}
  // expected-note@-2{{Dereference of null pointer}}
  (void) i;
  (void) x;
}

} // namespace NestedRegionTrack

namespace NestedElementRegionTrack {
struct N {
  int *arr[2];
};

struct S {
  N n;
};

void NestedElementRegionTrack() {
  int *x = nullptr, *y = nullptr;
  // expected-note@-1{{'y' initialized to a null pointer value}}

  S a{{x, y}};
  // expected-note@-1{{Initializing to a null pointer value}}
  // expected-note@-2{{'a' initialized here}}

  S b = a; // expected-note{{Storing null pointer value}}

  int i = *b.n.arr[1];
  // expected-warning@-1{{Dereference of null pointer}}
  // expected-note@-2{{Dereference of null pointer}}
  (void) i;
}

} // namespace NestedElementRegionTrack
