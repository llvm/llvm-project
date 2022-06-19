// RUN: %check_clang_tidy %s bugprone-suspicious-memory-comparison %t \
// RUN: -- -- -target x86_64-unknown-unknown -std=c99

typedef __SIZE_TYPE__ size_t;
int memcmp(const void *lhs, const void *rhs, size_t count);

// Examples from cert rule exp42-c

struct S {
  char c;
  int i;
  char buffer[13];
};

void exp42_c_noncompliant(const struct S *left, const struct S *right) {
  if ((left && right) && (0 == memcmp(left, right, sizeof(struct S)))) {
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: comparing object representation of type 'struct S' which does not have a unique object representation; consider comparing the members of the object manually
  }
}

void exp42_c_compliant(const struct S *left, const struct S *right) {
  if ((left && right) && (left->c == right->c) && (left->i == right->i) &&
      (0 == memcmp(left->buffer, right->buffer, 13))) {
  }
}

#pragma pack(push, 1)
struct Packed_S {
  char c;
  int i;
  char buffer[13];
};
#pragma pack(pop)

void compliant_packed(const struct Packed_S *left,
                      const struct Packed_S *right) {
  if ((left && right) && (0 == memcmp(left, right, sizeof(struct Packed_S)))) {
    // no-warning
  }
}

// Examples from cert rule flp37-c

struct S2 {
  int i;
  float f;
};

int flp37_c_noncompliant(const struct S2 *s1, const struct S2 *s2) {
  if (!s1 && !s2)
    return 1;
  else if (!s1 || !s2)
    return 0;
  return 0 == memcmp(s1, s2, sizeof(struct S2));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: comparing object representation of type 'struct S2' which does not have a unique object representation; consider comparing the members of the object manually
}

int flp37_c_compliant(const struct S2 *s1, const struct S2 *s2) {
  if (!s1 && !s2)
    return 1;
  else if (!s1 || !s2)
    return 0;
  return s1->i == s2->i && s1->f == s2->f;
  // no-warning
}

void Test_Float(void) {
  float a, b;
  memcmp(&a, &b, sizeof(float));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'float' which does not have a unique object representation; consider comparing the values manually
}

void TestArray_Float(void) {
  float a[3], b[3];
  memcmp(a, b, sizeof(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'float' which does not have a unique object representation; consider comparing the values manually
}

struct PredeclaredType;

void Test_PredeclaredType(const struct PredeclaredType *lhs,
                          const struct PredeclaredType *rhs) {
  memcmp(lhs, rhs, 1); // no-warning: predeclared type
}

struct NoPadding {
  int x;
  int y;
};

void Test_NoPadding(void) {
  struct NoPadding a, b;
  memcmp(&a, &b, sizeof(struct NoPadding));
}

void TestArray_NoPadding(void) {
  struct NoPadding a[3], b[3];
  memcmp(a, b, 3 * sizeof(struct NoPadding));
}

struct TrailingPadding {
  int i;
  char c;
};

void Test_TrailingPadding(void) {
  struct TrailingPadding a, b;
  memcmp(&a, &b, sizeof(struct TrailingPadding));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct TrailingPadding' which does not have a unique object representation; consider comparing the members of the object manually
  memcmp(&a, &b, sizeof(int)); // no-warning: not comparing entire object
  memcmp(&a, &b, 2 * sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct TrailingPadding' which does not have a unique object representation; consider comparing the members of the object manually
}

struct TrailingPadding2 {
  int i[2];
  char c;
};

void Test_TrailingPadding2(void) {
  struct TrailingPadding2 a, b;
  memcmp(&a, &b, 2 * sizeof(int)); // no-warning: not comparing entire object
  memcmp(&a, &b, sizeof(struct TrailingPadding2));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct TrailingPadding2' which does not have a unique object representation; consider comparing the members of the object manually
}

void Test_UnknownCount(size_t count) {
  struct TrailingPadding a, b;
  memcmp(&a, &b, count); // no-warning: unknown count value
}

void Test_ExplicitVoidCast(void) {
  struct TrailingPadding a, b;
  memcmp((void *)&a, (void *)&b,
         sizeof(struct TrailingPadding)); // no-warning: explicit cast
}

void TestArray_TrailingPadding(void) {
  struct TrailingPadding a[3], b[3];
  memcmp(a, b, 3 * sizeof(struct TrailingPadding));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct TrailingPadding' which does not have a unique object representation; consider comparing the members of the object manually
}

struct InnerPadding {
  char c;
  int i;
};

void Test_InnerPadding(void) {
  struct InnerPadding a, b;
  memcmp(&a, &b, sizeof(struct InnerPadding));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct InnerPadding' which does not have a unique object representation; consider comparing the members of the object manually
}

struct Bitfield_TrailingPaddingBytes {
  int x : 10;
  int y : 6;
};

void Test_Bitfield_TrailingPaddingBytes(void) {
  struct Bitfield_TrailingPaddingBytes a, b;
  memcmp(&a, &b, sizeof(struct S));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct Bitfield_TrailingPaddingBytes' which does not have a unique object representation; consider comparing the members of the object manually
}

struct Bitfield_TrailingPaddingBits {
  int x : 10;
  int y : 20;
};

void Test_Bitfield_TrailingPaddingBits(void) {
  struct Bitfield_TrailingPaddingBits a, b;
  memcmp(&a, &b, sizeof(struct Bitfield_TrailingPaddingBits));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct Bitfield_TrailingPaddingBits' which does not have a unique object representation; consider comparing the members of the object manually
}

struct Bitfield_InnerPaddingBits {
  char x : 2;
  int : 0;
  char y : 8;
};

void Test_Bitfield_InnerPaddingBits(void) {
  struct Bitfield_InnerPaddingBits a, b;
  memcmp(&a, &b, sizeof(struct Bitfield_InnerPaddingBits));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct Bitfield_InnerPaddingBits' which does not have a unique object representation; consider comparing the members of the object manually
}

struct Bitfield_NoPadding {
  int i : 10;
  int j : 10;
  int k : 10;
  int l : 2;
};
_Static_assert(sizeof(struct Bitfield_NoPadding) == sizeof(int),
               "Bit-fields should line up perfectly");

void Test_Bitfield_NoPadding(void) {
  struct Bitfield_NoPadding a, b;
  memcmp(&a, &b, sizeof(struct Bitfield_NoPadding)); // no-warning
}

struct Bitfield_TrailingUnnamed {
  int i[2];
  int : 0;
};

void Bitfield_TrailingUnnamed(void) {
  struct Bitfield_TrailingUnnamed a, b;
  memcmp(&a, &b, 2 * sizeof(int));                         // no-warning
  memcmp(&a, &b, sizeof(struct Bitfield_TrailingUnnamed)); // no-warning
}

struct PaddingAfterUnion {
  union {
    unsigned short a;
    short b;
  } x;

  int y;
};

void Test_PaddingAfterUnion(void) {
  struct PaddingAfterUnion a, b;
  memcmp(&a, &b, sizeof(struct PaddingAfterUnion));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct PaddingAfterUnion' which does not have a unique object representation; consider comparing the members of the object manually
}

struct Union_NoPadding {
  union {
    int a;
    unsigned int b;
  } x;

  int y;
};

void Test_Union_NoPadding(void) {
  struct Union_NoPadding a, b;
  memcmp(&a, &b, 2 * sizeof(int));
  memcmp(&a, &b, sizeof(struct Union_NoPadding));
}

union UnionWithPaddingInNestedStruct {
  int i;

  struct {
    int i;
    char c;
  } x;
};

void Test_UnionWithPaddingInNestedStruct(void) {
  union UnionWithPaddingInNestedStruct a, b;
  memcmp(&a, &b, sizeof(union UnionWithPaddingInNestedStruct));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'union UnionWithPaddingInNestedStruct' which does not have a unique object representation; consider comparing the members of the object manually
}

struct PaddingInNested {
  struct TrailingPadding x;
  char y;
};

void Test_PaddingInNested(void) {
  struct PaddingInNested a, b;
  memcmp(&a, &b, sizeof(struct PaddingInNested));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct PaddingInNested' which does not have a unique object representation; consider comparing the members of the object manually
}

struct PaddingAfterNested {
  struct {
    char a;
    char b;
  } x;
  int y;
};

void Test_PaddingAfterNested(void) {
  struct PaddingAfterNested a, b;
  memcmp(&a, &b, sizeof(struct PaddingAfterNested));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct PaddingAfterNested' which does not have a unique object representation; consider comparing the members of the object manually
}

struct AtomicMember {
  _Atomic(int) x;
};

void Test_AtomicMember(void) {
  // FIXME: this is a false positive as the list of objects with unique object
  // representations is incomplete.
  struct AtomicMember a, b;
  memcmp(&a, &b, sizeof(struct AtomicMember));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'struct AtomicMember' which does not have a unique object representation; consider comparing the members of the object manually
}
