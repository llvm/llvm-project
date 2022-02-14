// FIXME -Wno-aix-compat added temporarily while the diagnostic is being
// refined.
// RUN: %clang_analyze_cc1 -verify -Wno-aix-compat %s \
// RUN:   -analyzer-checker=optin.performance \
// RUN:   -analyzer-config optin.performance.Padding:AllowedPad=2

// RUN: not %clang_analyze_cc1 -verify -Wno-aix-compat %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=optin.performance.Padding \
// RUN:   -analyzer-config optin.performance.Padding:AllowedPad=-10 \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-PAD-NEGATIVE-VALUE

// CHECK-PAD-NEGATIVE-VALUE: (frontend): invalid input for checker option
// CHECK-PAD-NEGATIVE-VALUE-SAME: 'optin.performance.Padding:AllowedPad', that
// CHECK-PAD-NEGATIVE-VALUE-SAME: expects a non-negative value

#if __has_include(<stdalign.h>)
#include <stdalign.h>
#endif

#if __has_include(<stdalign.h>) || defined(__cplusplus)
// expected-warning@+1{{Excessive padding in 'struct FieldAttrAlign' (6 padding}}
struct FieldAttrAlign {
  char c1;
  alignas(4) int i;
  char c2;
};

// expected-warning@+1{{Excessive padding in 'struct FieldAttrOverAlign' (10 padding}}
struct FieldAttrOverAlign {
  char c1;
  alignas(8) int i;
  char c2;
};

#endif // __has_include(<stdalign.h>) || defined(__cplusplus)

// Re-ordering members of these structs won't reduce padding, so don't warn
struct LeadingChar { // no-warning
  char c;
  int i;
};

struct TrailingChar { // no-warning
  int i;
  char c;
};

struct Helpless { // no-warning
  struct TrailingChar i1;
  struct LeadingChar i2;
  char c;
};

#pragma pack(push)
#pragma pack(1)
struct SquishedIntSandwich { // no-warning
  char c1;
  int i;
  char c2;
};
#pragma pack(pop)

// Re-ordering members of these structs will reduce padding, so warn
struct IntSandwich { // expected-warning{{Excessive padding in 'struct IntSandwich'}}
  char c1;
  int i;
  char c2;
};

struct TurDuckHen { // expected-warning{{Excessive padding in 'struct TurDuckHen'}}
  char c1;
  struct IntSandwich i;
  char c2;
};

#pragma pack(push)
#pragma pack(2)
struct SmallIntSandwich { // expected-warning{{Excessive padding in 'struct SmallIntSandwich'}}
  char c1;
  int i1;
  char c2;
  int i2;
  char c3;
  int i3;
  char c4;
};
#pragma pack(pop)

union SomeUnion { // no-warning
  char c;
  short s;
  int i;
};

struct HoldsAUnion { // expected-warning{{Excessive padding in 'struct HoldsAUnion'}}
  char c1;
  union SomeUnion u;
  char c2;
};

struct BigCharArray { // no-warning
  char c[129];
};

struct SmallCharArray { // no-warning
  char c[5];
};

struct MediumIntArray { // no-warning
  int i[5];
};

struct LargeSizeToSmallSize { // expected-warning{{Excessive padding in 'struct LargeSizeToSmallSize'}}
  struct BigCharArray b;
  struct MediumIntArray m;
  struct SmallCharArray s;
};

struct LargeAlignToSmallAlign { // no-warning
  struct MediumIntArray m;
  struct BigCharArray b;
  struct SmallCharArray s;
};

// Currently ignoring VLA padding problems.  Still need to make sure we don't
// choke on VLAs though
struct HoldsVLA { // no-warning
  char c1;
  int x;
  char c2;
  int vla[];
};

// Currently ignoring bitfield padding problems.  Still need to make sure we
// don't choke on bitfields though
struct HoldsBitfield { // no-warning
  char c1;
  int x;
  char c2;
  unsigned char b1 : 3;
  unsigned char b2 : 3;
  unsigned char b3 : 2;
};

typedef struct { // expected-warning{{Excessive padding in 'TypedefSandwich'}}
  char c1;
  int i;
  char c2;
} TypedefSandwich;

// expected-warning@+1{{Excessive padding in 'struct StructAttrAlign' (10 padding}}
struct StructAttrAlign {
  char c1;
  int i;
  char c2;
} __attribute__((aligned(8)));

struct CorrectOverlyAlignedChar { // no-warning
  char c __attribute__((aligned(4096)));
  char c1;
  int x1;
  char c2;
  int x2;
  char c3;
};

struct OverlyAlignedChar { // expected-warning{{Excessive padding in 'struct OverlyAlignedChar'}}
  char c1;
  int x;
  char c2;
  char c __attribute__((aligned(4096)));
};

struct HoldsOverlyAlignedChar { // expected-warning{{Excessive padding in 'struct HoldsOverlyAlignedChar'}}
  char c1;
  struct OverlyAlignedChar o;
  char c2;
};

void internalStructFunc(void) {
  struct X { // expected-warning{{Excessive padding in 'struct X'}}
    char c1;
    int t;
    char c2;
  };
  struct X obj;
}

void typedefStructFunc(void) {
  typedef struct { // expected-warning{{Excessive padding in 'S'}}
    char c1;
    int t;
    char c2;
  } S;
  S obj;
}

void anonStructFunc(void) {
  struct { // expected-warning{{Excessive padding in 'struct (unnamed}}
    char c1;
    int t;
    char c2;
  } obj;
}

struct CorrectDefaultAttrAlign { // no-warning
  long long i;
  char c1;
  char c2;
} __attribute__((aligned));

struct TooSmallShortSandwich { // no-warning
  char c1;
  short s;
  char c2;
};

// expected-warning@+1{{Excessive padding in 'struct SmallArrayShortSandwich'}}
struct SmallArrayShortSandwich {
  char c1;
  short s;
  char c2;
} ShortArray[20];

// expected-warning@+1{{Excessive padding in 'struct SmallArrayInFunc'}}
struct SmallArrayInFunc {
  char c1;
  short s;
  char c2;
};

void arrayHolder(void) {
  struct SmallArrayInFunc Arr[15];
}

// xxxexpected-warning@+1{{Excessive padding in 'struct SmallArrayInStruct'}}
struct SmallArrayInStruct {
  char c1;
  short s;
  char c2;
};

struct HoldsSmallArray {
  struct SmallArrayInStruct Field[20];
} HoldsSmallArrayElt;

void nestedPadding(void) {
  struct HoldsSmallArray Arr[15];
}
