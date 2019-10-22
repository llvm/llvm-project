// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core,unix.Malloc,debug.ExprInspection %s -analyzer-config eagerly-assume=false -verify

extern void clang_analyzer_eval(bool);
extern void clang_analyzer_warnIfReached();
extern "C" char *strdup(const char *s);

namespace PR14054_reduced {
  struct Definition;
  struct ParseNode {
    union {
      Definition *lexdef;
      ParseNode *data;
    } pn_u;
  };
  struct Definition : public ParseNode { };

  void CloneParseTree(ParseNode *opn, ParseNode *pn,  ParseNode *x) {
    // This used to cause an assertion failure because:
    // 1. The implicit operator= for unions assigns all members of the union,
    //    not just the active one (b/c there's no way to know which is active).
    // 2. RegionStore dutifully stored all the variants at the same offset;
    //    the last one won.
    // 3. We asked for the value of the first variant but got back a conjured
    //    symbol for the second variant.
    // 4. We ended up trying to add a base cast to a region of the wrong type.
    //
    // Now (at the time this test was added), we instead treat all variants of
    // a union as different offsets, but only allow one to be active at a time.
    *pn = *opn;
    x = pn->pn_u.lexdef->pn_u.lexdef;
  }
}

namespace PR14054_original {
  struct Definition;
  struct ParseNode {
    union {
      struct {
        union {}; // expected-warning {{does not declare anything}}
        Definition *lexdef;
      } name;
      class {
        int *target;
        ParseNode *data;
      } xmlpi;
    } pn_u;
  };
  struct Definition : public ParseNode { };

  void CloneParseTree(ParseNode *opn, ParseNode *pn,  ParseNode *x) {
    pn->pn_u = opn->pn_u;
    x = pn->pn_u.name.lexdef->pn_u.name.lexdef;
  }
}

namespace PR17596 {
  union IntOrString {
    int i;
    char *s;
  };

  extern void process(IntOrString);

  void test() {
    IntOrString uu;
    uu.s = strdup("");
    process(uu);
  }

  void testPositive() {
    IntOrString uu;
    uu.s = strdup("");
  } // expected-warning{{leak}}

  void testCopy() {
    IntOrString uu;
    uu.i = 4;
    clang_analyzer_eval(uu.i == 4); // expected-warning{{TRUE}}

    IntOrString vv;
    vv.i = 5;
    uu = vv;
    clang_analyzer_eval(uu.i == 5); // expected-warning{{TRUE}}
  }

  void testInvalidation() {
    IntOrString uu;
    uu.s = strdup("");

    IntOrString vv;
    char str[] = "abc";
    vv.s = str;

    uu = vv;
  } // expected-warning{{leak}}

  void testIndirectInvalidation() {
    IntOrString uu;
    char str[] = "abc";
    uu.s = str;

    clang_analyzer_eval(uu.s[0] == 'a'); // expected-warning{{TRUE}}

    process(uu);
    clang_analyzer_eval(uu.s[0] == 'a'); // expected-warning{{UNKNOWN}}
  }
}

namespace assume_union_contents {
union U {
  int x;
};

U get();

void test() {
  U u = get();
  int y = 0;
  if (u.x)
    y = 1;
  if (u.x)
    y = 1 / y; // no-warning
}
} // end namespace assume_union_contents

namespace pr37688_deleted_union_destructor {
struct S { ~S(); };
struct A {
  ~A() noexcept {}
  union {
    struct {
      S s;
    } ss;
  };
};
void foo() {
  A a;
} // no-crash
void bar() {
  foo();
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
} // end namespace pr37688_deleted_union_destructor
