// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace PR23186 {
decltype(ned);  // expected-error {{use of undeclared identifier 'ned'}}
// The code below was triggering an UNREACHABLE in ASTContext::getTypeInfoImpl
// once the above code failed to recover properly after making the bogus
// correction of 'ned' to 'new'.
template <typename>
struct S {
  enum { V };
  void f() {
    switch (0)
    case V:
      ;
  }
};
}

namespace PR23140 {
auto lneed = gned.*[] {};  // expected-error-re {{use of undeclared identifier 'gned'{{$}}}}

void test(int aaa, int bbb, int thisvar) {
  int thatval = aaa * (bbb + thatvar);  // expected-error {{use of undeclared identifier 'thatvar'; did you mean 'thatval'}} \
                                           expected-note {{'thatval' declared here}}
}
}

namespace PR18854 {
void f() {
  for (auto&& x : e) {  // expected-error-re {{use of undeclared identifier 'e'{{$}}}}
    auto Functor = [x]() {};
    long Alignment = __alignof__(Functor);
  }
}
}

namespace NewTypoExprFromResolvingTypoAmbiguity {
struct A {
  void Swap(A *other);
};

struct pair {
  int first;
  A *second;
};

struct map {
public:
  void swap(map &x);
  pair find(int x);
};

void run(A *annotations) {
  map new_annotations;

  auto &annotation = *annotations;
  auto new_it = new_annotations.find(5);
  auto &new_anotation = new_it.second;
  new_annotation->Swap(&annotation);  // expected-error {{use of undeclared identifier 'new_annotation'}}
}
}
