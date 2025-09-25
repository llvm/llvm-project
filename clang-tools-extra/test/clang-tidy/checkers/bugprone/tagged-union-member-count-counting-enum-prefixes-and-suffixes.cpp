// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: false, \
// RUN:     bugprone-tagged-union-member-count.EnableCountingEnumHeuristic: true, \
// RUN:     bugprone-tagged-union-member-count.CountingEnumSuffixes: "count", \
// RUN:     bugprone-tagged-union-member-count.CountingEnumPrefixes: "last", \
// RUN:  }}' --

union Union3 {
  short *Shorts;
  int *Ints;
  float *Floats;
};

union Union4 {
  short *Shorts;
  double *Doubles;
  int *Ints;
  float *Floats;
};

// The heuristic only considers the last enum constant
// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionPrefixAndSuffixMatch {
  enum {
    tags1,
    tags2,
    tagscount,
    lasttags
  } Kind;
  Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (3) than tags (2)
struct TaggedUnionOnlyPrefixMatch {
  enum { 
    prefixtag1,
    prefixtag2,
    lastprefixtag
  } Kind;
  Union3 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (3) than tags (2)
struct TaggedUnionOnlySuffixMatch {
  enum {
    suffixtag1,
    suffixtag2,
    suffixtagcount
  } Kind;
  Union3 Data;
};
