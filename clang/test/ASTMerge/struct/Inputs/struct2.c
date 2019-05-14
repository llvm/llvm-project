// Matches
struct S0 {
  int field1;
  float field2;
};

struct S0 x0;

// Mismatch in field type
struct S1 {
  int field1;
  float field2;
};

struct S1 x1;

// Mismatch in tag kind.
union S2 { int i; float f; } x2;

// Missing fields
struct S3 { int i; float f; } x3;

// Extra fields
struct S4 { int i; float f; } x4;

// Bit-field matches
struct S5 { int i : 8; unsigned j : 8; } x5;

// Bit-field mismatch
struct S6 { int i : 8; unsigned j; } x6;

// Bit-field mismatch
struct S7 { int i : 8; unsigned j : 16; } x7;

// Incomplete type
struct S8 { int i; float f; } *x8;

// Incomplete type
struct S9 *x9;

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
  struct DeeperError { int i; float f; } *Deeper;
} xDeep;

// Matches
struct {
  int i;
  float f;
} x11;

// Matches
typedef struct {
  int i;
  float f;
} S12;

S12 x12;

// Mismatch
typedef struct {
  int i; // Mismatch here.
  float f;
} S13;

S13 x13;

// Matches
struct Unnamed {
  union {
    struct {
      int i;
    } S;
    struct {
      float i;
    } R;
  } U;
} x14;

// Matches
struct DeepUnnamed {
  union {
    union {
      struct {
        long i;
      } S;
      struct {
        int i;
      } R;
    } U1;
    union {
      struct {
        long i;
      } S;
      struct {
        float i;
      } T;
    } U2;
  } U;
  struct {
    long i;
  } V;
} x15;

// Mismatch due to unnamed struct used internally
struct DeepUnnamedError {
  union {
    union {
      struct {
        long i;
      } S;
      struct {
        int i;
      } R;
    } U1;
    union {
      struct {
        float i; // Mismatch here.
      } S;
      struct {
        float i;
      } T;
    } U2;
  } U;
  struct {
    long i;
  } V;
} x16;
