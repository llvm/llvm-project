// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: false, \
// RUN:     bugprone-tagged-union-member-count.EnableCountingEnumHeuristic: true, \
// RUN:     bugprone-tagged-union-member-count.CountingEnumSuffixes: "count", \
// RUN:     bugprone-tagged-union-member-count.CountingEnumPrefixes: "last", \
// RUN:  }}' --

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (3) than tags (2)
struct IncorrectBecauseHeuristicIsEnabledPrefixCase {
  enum {
    tags1,
    tags2,
    lasttag,
  } Tags;
  union {
    char A;
    short B;
    int C;
  } Data;
};

struct CorrectBecauseHeuristicIsEnabledPrefixCase { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
    lasttag,
  } Tags;
  union {
    int A;
    int B;
    int C;
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (3) than tags (2)
struct IncorrectBecauseHeuristicIsEnabledSuffixCase {
  enum {
    tags1,
    tags2,
    tags_count,
  } Tags;
  union {
    char A;
    short B;
    int C;
  } Data;
};

struct CorrectBecauseHeuristicIsEnabledSuffixCase { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
    tags_count,
  } Tags;
  union {
    int A;
    int B;
    int C;
  } Data;
};

union Union4 {
  short *Shorts;
  double *Doubles;
  int *Ints;
  float *Floats;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct CountingEnumCaseInsensitivityTest1 { 
  enum {
    node_type_loop,
    node_type_branch,
    node_type_function,
    node_type_count,
  } Kind;
  union Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct CountingEnumCaseInsensitivityTest2 { 
  enum {
    NODE_TYPE_LOOP,
    NODE_TYPE_BRANCH,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_COUNT,
  } Kind;
  union Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TagWhereCountingEnumIsAliased {
  enum {
    tag_alias_counter1 = 1,
    tag_alias_counter2 = 2,
    tag_alias_counter3 = 3,
    tag_alias_other_count = 3,
  } Kind;
  union {
    char C;
    short S;
    int I;
    long L;
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (2)
struct TagWithCountingEnumButOtherValueIsAliased {
  enum {
    tag_alias_other1 = 1,
    tag_alias_other2 = 1,
    tag_alias_other3 = 3,
    tag_alias_other_count = 2,
  } Kind;
  union {
    char C;
    short S;
    int I;
    long L;
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TagWhereCounterIsTheSmallest {
  enum {
    tag_large1 = 1000,
    tag_large2 = 1001,
    tag_large3 = 1002,
    tag_large_count = 3,
  } Kind;
  union {
    char C;
    short S;
    int I;
    long L;
  } Data;
};

// No warnings expected, only the last enum constant can be a counting enum constant
struct TagWhereCounterLikeNameIsNotLast {
  enum {
    kind_count,
    kind2,
    last_kind1,
    kind3,
  } Kind;
  union {
    char C;
    short S;
    int I;
    long L;
  } Data;
};
