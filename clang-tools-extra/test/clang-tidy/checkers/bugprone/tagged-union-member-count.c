// RUN: %check_clang_tidy %s bugprone-tagged-union-member-count %t

typedef enum Tags3 {
  tags3_1,
  tags3_2,
  tags3_3,
} Tags3;

typedef enum Tags4 {
  tags4_1,
  tags4_2,
  tags4_3,
  tags4_4,
} Tags4;

typedef union Union3 {
  short *Shorts;
  int *Ints;
  float *Floats;
} Union3;

typedef union Union4 {
  short *Shorts;
  double *Doubles;
  int *Ints;
  float *Floats;
} Union4;

// It is not obvious which enum is the tag for the union.
struct maybeTaggedUnion1 { // No warnings expected.
  enum Tags3 TagA;
  enum Tags4 TagB;
  union Union4 Data;
};

// It is not obvious which union does the tag belong to.
struct maybeTaggedUnion2 { // No warnings expected.
  enum Tags3 Tag;
  union Union3 DataB;
  union Union3 DataA;
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

// No warnings expected, because LastATag is just an alias
struct TaggedUnionWithAliasedEnumConstant {
  enum {
    ATag1,
    ATag2,
    ATag3,
    LastATag = ATag3,
  } Tag;
  union {
    float F;
    int *Ints;
    char Key[8];
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithPredefinedTagAndPredefinedUnion {
  enum Tags3 Tag;
    union Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
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

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithInlineTagAndPredefinedUnion { 
  enum {
    TaggedUnion7tag1,
    TaggedUnion7tag2,
    TaggedUnion7tag3,
  } Tag;
  union Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
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

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructNesting { 
  enum Tags3 Tag;
  union {
    float F;
    int I;
    long L;
    // CHECK-MESSAGES: :[[@LINE+1]]:12: warning: tagged union has more data members (4) than tags (3)
    struct innerdecl { 
      enum Tags3 Tag;
      union Union4 Data;
    } Inner; 
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithTypedefedTagAndTypedefedUnion { 
  Tags3 Tag;
  Union4 Data;
};

#define DECLARE_TAGGED_UNION_STRUCT(Tag, Union, Name)\
struct Name {\
  Tag Kind;\
  Union Data;\
}

// CHECK-MESSAGES: :[[@LINE+1]]:44: warning: tagged union has more data members (4) than tags (3)
DECLARE_TAGGED_UNION_STRUCT(Tags3, Union4, TaggedUnionStructFromMacro);
