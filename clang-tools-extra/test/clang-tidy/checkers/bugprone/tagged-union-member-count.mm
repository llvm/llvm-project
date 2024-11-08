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

enum class Classtags3 {
  classtags3_1,
  classtags3_2,
  classtags3_3,
};

enum class Typedtags3 : unsigned int {
  typedtags3_1,
  typedtags3_2,
  typedtags3_3,
};

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
class MaybeTaggedUnion1 { // No warnings expected.
  enum Tags3 TagA;
  enum Tags4 TagB;
  union Union4 Data;
};

// It is not obvious which union does the tag belong to.
class MaybeTaggedUnion2 { // No warnings expected.
  enum Tags3 Tag;
  union Union3 DataB;
  union Union3 DataA;
};

// It is not obvious which union does the tag belong to.
class MaybeTaggedUnion3 { // No warnings expected.
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

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassPredefinedTagAndPredefinedUnion {
  enum Tags3 Tag;
    union Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassPredefinedTagAndInlineUnion {
  enum Tags3 Tag;
    union {
    int *Ints;
    char Characters[13];
    class {
      double Re;
      double Im;
    } Complex;
    long L;
    } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassInlineTagAndPredefinedUnion { 
  enum {
    tag1,
    tag2,
    tag3,
  } Tag;
  union Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassInlineTagAndInlineUnion { 
  enum {
    tag1,
    tag2,
    tag3,
  } Tag;
  union {
    int *Ints;
    char Characters[13];
    class {
      double Re;
      double Im;
    } Complex;
    long L;
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassWithNestedTaggedUnionClass { 
  enum Tags3 Tag;
  union {
    float F;
    int I;
    long L;
    // CHECK-MESSAGES: :[[@LINE+1]]:11: warning: tagged union has more data members (4) than tags (3)
    class Innerdecl { 
      enum Tags3 Tag;
      union Union4 Data;
    } Inner; 
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassWithTypedefedTag { 
  Tags3 Tag;
  Union4 Data;
}; 

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithEnumClass { 
  enum Classtags3 Tag;
  Union4 Data;
}; 

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClasswithEnumClass { 
  enum Classtags3 Tag;
  Union4 Data;
}; 

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithTypedEnum {
  Typedtags3 Tag;
  Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassWithTypedEnum {
  Typedtags3 Tag;
  Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct AnonymousTaggedUnionStruct {
  Tags3 Tag;
  union {
    char A;
    short B;
    int C;
    long D;
  };
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassWithAnonymousUnion {
  Tags3 Tag;
  union {
    char A;
    short B;
    int C;
    long D;
  };
};

namespace testnamespace {

enum Tags3 {
  tags3_1,
  tags3_2,
  tags3_3,
};

union Union4 {
  short *Shorts;
  double *Doubles;
  int *Ints;
  float *Floats;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructInNamespace {
  Tags3 Tags;
  Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassInNamespace {
  Tags3 Tags;
  Union4 Data;
};

} // namespace testnamespace

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (4) than tags (3)
struct TaggedUnionStructWithNamespacedTagAndUnion {
  testnamespace::Tags3 Tags;
  testnamespace::Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: tagged union has more data members (4) than tags (3)
class TaggedUnionClassWithNamespacedTagAndUnion {
  testnamespace::Tags3 Tags;
  testnamespace::Union4 Data;
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: tagged union has more data members (4) than tags (3)
template <typename Tag, typename Union>
struct TemplatedStructWithNamespacedTagAndUnion {
  Tag Kind;
  Union Data;
};

TemplatedStructWithNamespacedTagAndUnion<testnamespace::Union4, testnamespace::Tags3> TemplatedStruct3;

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: tagged union has more data members (4) than tags (3)
template <typename Tag, typename Union>
class TemplatedClassWithNamespacedTagAndUnion {
  Tag Kind;
  Union Data;
};

TemplatedClassWithNamespacedTagAndUnion<testnamespace::Union4, testnamespace::Tags3> TemplatedClass3;

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: tagged union has more data members (4) than tags (3)
template <typename Tag, typename Union>
struct TemplatedStruct {
  Tag Kind;
  Union Data;
};

TemplatedStruct<Tags3, Union3> TemplatedStruct1; // No warning expected
TemplatedStruct<Tags3, Union4> TemplatedStruct2;

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: tagged union has more data members (4) than tags (3)
template <typename Tag, typename Union>
class TemplatedClass {
  Tag Kind;
  Union Data;
};

TemplatedClass<Tags3, Union3> TemplatedClass1; // No warning expected
TemplatedClass<Tags3, Union4> TemplatedClass2;

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: tagged union has more data members (4) than tags (3)
template <typename T>
struct TemplatedStructButTaggedUnionPartIsNotTemplated {
  Tags3 Kind;
  Union4 Data;
  T SomethingElse;
};

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: tagged union has more data members (4) than tags (3)
template <typename T>
class TemplatedClassButTaggedUnionPartIsNotTemplated {
  Tags3 Kind;
  Union4 Data;
  T SomethingElse;
};

#define DECLARE_TAGGED_UNION_STRUCT(Tag, Union, Name)\
struct Name {\
  Tag Kind;\
  Union Data;\
}

// CHECK-MESSAGES: :[[@LINE+1]]:44: warning: tagged union has more data members (4) than tags (3)
DECLARE_TAGGED_UNION_STRUCT(Tags3, Union4, TaggedUnionStructFromMacro);

#define DECLARE_TAGGED_UNION_CLASS(Tag, Union, Name)\
class Name {\
  Tag Kind;\
  Union Data;\
}

// CHECK-MESSAGES: :[[@LINE+1]]:43: warning: tagged union has more data members (4) than tags (3)
DECLARE_TAGGED_UNION_CLASS(Tags3, Union4, TaggedUnionClassFromMacro);

// Lambdas implicitly compile down to an unnamed CXXRecordDecl and if they have captures,
// then those become unnamed fields.
void DoNotMatchLambdas() {
    enum {
		A
	} e;
    union {
		long A;
		char B;
	} u;
    auto L = [e, u] () {};
}
