// RUN: %clang_cc1 -analyze -analyzer-checker=core %s -verify
// expected-no-diagnostics

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
        union {};
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
