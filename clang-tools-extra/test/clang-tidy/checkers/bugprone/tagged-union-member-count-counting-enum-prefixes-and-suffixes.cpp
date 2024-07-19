// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictModeIsEnabled: 0, \
// RUN:     bugprone-tagged-union-member-count.CountingEnumHeuristicIsEnabled: 1, \
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

// No warning expected, because Kind has multiple counting enum candidates,
// therefore the enum count is left unchanged.
struct TaggedUnionPrefixAndSuffixMatch {
  enum {
    tags1,
    tags2,
    tagscount,
    lasttags
  } Kind;
  Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (3) than tags (2)
struct TaggedUnionOnlyPrefixMatch {
  enum { 
    prefixtag1,
    prefixtag2,
    lastprefixtag
  } Kind;
  Union3 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (3) than tags (2)
struct TaggedUnionOnlySuffixMatch {
  enum {
    suffixtag1,
    suffixtag2,
    suffixtagcount
  } Kind;
  Union3 Data;
};

