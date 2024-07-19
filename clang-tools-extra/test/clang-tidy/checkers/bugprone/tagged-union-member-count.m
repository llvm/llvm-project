// RUN: %check_clang_tidy %s bugprone-tagged-union-member-count %t

typedef enum Tags3 {
  tags3_1,
  tags3_2,
  tags3_3,
} tags3;

typedef enum Tags4 {
  tags4_1,
  tags4_2,
  tags4_3,
  tags4_4,
} tags4;

typedef union union3 {
  short *Shorts;
  int *Ints;
  float *Floats;
} union3;

typedef union union4 {
  short *Shorts;
  double *Doubles;
  int *Ints;
  float *Floats;
} union4;

// It is not obvious which enum is the tag for the union.
struct maybeTaggedUnion1 { // No warnings expected.
  enum Tags3 TagA;
  enum Tags4 TagB;
  union union4 Data;
};

// It is not obvious which union does the tag belong to.
struct maybeTaggedUnion2 { // No warnings expected.
  enum Tags3 Tag;
  union union3 DataB;
  union union3 DataA;
};

// It is not obvious which union does the tag belong to.
struct maybeTaggedUnion3 { // No warnings expected.
  enum Tags3 Tag;
  union {
    int I1;
    int I2;
    int I3;
  };
  union {
    float F1;
    float F2;
    float F3;
  };
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithPredefinedTagAndPredefinedUnion {
  enum Tags3 Tag;
    union union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithPredefinedTagAndInlineUnion {
  enum Tags3 Tag;
    union {
    int *Ints;
    char Characters[13];
    struct {
      double Re;
      double Im;
    } Complex;
    long L;
    } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithInlineTagAndPredefinedUnion { 
  enum {
    TaggedUnion7tag1,
    TaggedUnion7tag2,
    TaggedUnion7tag3,
  } Tag;
  union union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithInlineTagAndInlineUnion { 
  enum {
    TaggedUnion8tag1,
    TaggedUnion8tag2,
    TaggedUnion8tag3,
  } Tag;
  union {
    int *Ints;
    char Characters[13];
    struct {
      double Re;
      double Im;
    } Complex;
    long L;
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct TaggedUnionStructNesting { 
  enum Tags3 Tag;
  union {
    float F;
    int I;
    long L;
    // CHECK-MESSAGES: :[[@LINE+1]]:12: warning: Tagged union has more data members (4) than tags (3)
    struct innerdecl { 
      enum Tags3 Tag;
      union union4 Data;
    } Inner; 
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct CountingEnumCaseInsensitivityTest1 { 
  enum {
    node_type_loop,
    node_type_branch,
    node_type_function,
    node_type_count,
  } Kind;
  union union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct CountingEnumCaseInsensitivityTest2 { 
  enum {
    NODE_TYPE_LOOP,
    NODE_TYPE_BRANCH,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_COUNT,
  } Kind;
  union union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithTypedefedTagAndTypedefedUnion { 
  tags3 Tag;
  union4 Data;
};
