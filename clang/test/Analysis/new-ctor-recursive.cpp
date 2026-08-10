// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus.NewDelete,cplusplus.NewDeleteLeaks,debug.ExprInspection -analyzer-config c++-allocator-inlining=true -std=c++11 -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);
void clang_analyzer_dump(int);

typedef __typeof__(sizeof(int)) size_t;

void *conjure();
void exit(int);

struct S;

S *global_s;

// Recursive operator kinda placement new.
void *operator new(size_t size, S *place);

enum class ConstructionKind : char {
  Garbage,
  Recursive
};

struct S {
public:
  int x;
  S(): x(1) {}
  S(int y): x(y) {}

  S(ConstructionKind k) {
    switch (k) {
    case ConstructionKind::Recursive: { // Call one more operator new 'r'ecursively.
      S *s = new (nullptr) S(5);
      x = s->x + 1;
      global_s = s;
      return;
    }
    case ConstructionKind::Garbage: {
      // Leaves garbage in 'x'.
    }
    }
  }
  ~S() {}
};

// Do not try this at home!
void *operator new(size_t size, S *place) {
  if (!place)
    return new S();
  return place;
}

void testThatCharConstructorIndeedYieldsGarbage() {
  S *s = new S(ConstructionKind::Garbage);
  clang_analyzer_eval(s->x == 0); // expected-warning{{The left operand of '==' is a garbage value [core.UndefinedBinaryOperatorResult]}}
  clang_analyzer_eval(s->x == 1);
  s->x += 1;
  delete s;
}


void testChainedOperatorNew() {
  S *s;
  // * Evaluate standard new.
  // * Evaluate constructor S(3).
  // * Bind value for standard new.
  // * Evaluate our custom new.
  // * Evaluate constructor S(Garbage).
  // * Bind value for our custom new.
  s = new (new S(3)) S(ConstructionKind::Garbage);
  clang_analyzer_eval(s->x == 3); // expected-warning{{TRUE}}
  // expected-warning@+9{{Potential leak of memory pointed to by 's'}}

  // * Evaluate standard new.
  // * Evaluate constructor S(Garbage).
  // * Bind value for standard new.
  // * Evaluate our custom new.
  // * Evaluate constructor S(4).
  // * Bind value for our custom new.
  s = new (new S(ConstructionKind::Garbage)) S(4);
  clang_analyzer_eval(s->x == 4); // expected-warning{{TRUE}}
  delete s;

  // -> Enter our custom new (nullptr).
  //   * Evaluate standard new.
  //   * Inline constructor S().
  //   * Bind value for standard new.
  // <- Exit our custom new (nullptr).
  // * Evaluate constructor S(Garbage).
  // * Bind value for our custom new.
  s = new (nullptr) S(ConstructionKind::Garbage);
  clang_analyzer_eval(s->x == 1); // expected-warning{{TRUE}}
  delete s;

  // -> Enter our custom new (nullptr).
  //   * Evaluate standard new.
  //   * Inline constructor S().
  //   * Bind value for standard new.
  // <- Exit our custom new (nullptr).
  // -> Enter constructor S(Recursive).
  //   -> Enter our custom new (nullptr).
  //     * Evaluate standard new.
  //     * Inline constructor S().
  //     * Bind value for standard new.
  //   <- Exit our custom new (nullptr).
  //   * Evaluate constructor S(5).
  //   * Bind value for our custom new (nullptr).
  //   * Assign that value to global_s.
  // <- Exit constructor S(Recursive).
  // * Bind value for our custom new (nullptr).
  global_s = nullptr;
  s = new (nullptr) S(ConstructionKind::Recursive);
  clang_analyzer_eval(global_s); // expected-warning{{TRUE}}
  clang_analyzer_eval(global_s->x == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(s->x == 6); // expected-warning{{TRUE}}
  delete s;
}
