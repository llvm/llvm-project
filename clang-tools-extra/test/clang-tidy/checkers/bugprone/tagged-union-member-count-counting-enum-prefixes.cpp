// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: false, \
// RUN:     bugprone-tagged-union-member-count.EnableCountingEnumHeuristic: true, \
// RUN:     bugprone-tagged-union-member-count.CountingEnumPrefixes: "maxsize;last", \
// RUN:  }}' --

union Union4 {
  short *Shorts;
  double *Doubles;
  int *Ints;
  float *Floats;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionWithMaxsizeAsCounterPrefix {
  enum {
    twc1,
    twc2,
    twc3,
    maxsizetwc,  
  } Kind;
  Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionWithLastAsCounterPrefix { 
  enum {
    twc11,
    twc22,
    twc33,
    lasttwc,
  } Kind;
  Union4 Data;
};
