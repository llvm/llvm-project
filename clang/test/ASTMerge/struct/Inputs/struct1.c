typedef int Int;
typedef float Float;

// Matches
struct S0 {
  Int field1;
  Float field2;
};

struct S0 x0;

// Mismatch in field type
struct S1 {
  Int field1;
  int field2;
};

struct S1 x1;

// Mismatch in tag kind.
struct S2 { int i; float f; } x2;

// Missing fields
struct S3 { int i; float f; double d; } x3;

// Extra fields
struct S4 { int i; } x4;

// Bit-field matches
struct S5 { int i : 8; unsigned j : 8; } x5;

// Bit-field mismatch
struct S6 { int i : 8; unsigned j : 8; } x6;

// Bit-field mismatch
struct S7 { int i : 8; unsigned j : 8; } x7;

// Incomplete type
struct S8 *x8;

// Incomplete type
struct S9 { int i; float f; } *x9;

// Incomplete type
struct S10 *x10;

// Matches
struct ListNode {
  int value;
  struct ListNode *Next;
} xList;

// Mismatch due to struct used internally
struct DeepError {
  int value;
  struct DeeperError { int i; int f; } *Deeper;
} xDeep;

// Matches
struct {
  Int i;
  float f;
} x11;

// Matches
typedef struct {
  Int i;
  float f;
} S12;

S12 x12;

// Mismatch
typedef struct {
  Float i; // Mismatch here.
  float f;
} S13;

S13 x13;
