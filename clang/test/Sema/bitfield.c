// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c11 -Wno-unused-value

enum e0; // expected-note{{forward declaration of 'enum e0'}}

struct a {
  int a : -1; // expected-error{{bit-field 'a' has negative width}}

  // rdar://6081627
  int b : 33; // expected-error{{width of bit-field 'b' (33 bits) exceeds width of its type (32 bits)}}

  int c : (1 + 0.25); // expected-error{{integer constant expression must have integer type}}
  int d : (int)(1 + 0.25); 

  // rdar://6138816
  int e : 0;  // expected-error {{bit-field 'e' has zero width}}

  float xx : 4;  // expected-error {{bit-field 'xx' has non-integral type}}

  // PR3607
  enum e0 f : 1; // expected-error {{field has incomplete type 'enum e0'}}
  
  int g : (_Bool)1;
  
  // PR4017  
  char : 10;      // expected-error {{width of anonymous bit-field (10 bits) exceeds width of its type (8 bits)}}
  unsigned : -2;  // expected-error {{anonymous bit-field has negative width (-2)}}
  float : 12;     // expected-error {{anonymous bit-field has non-integral type 'float'}}

  _Bool : 2;   // expected-error {{width of anonymous bit-field (2 bits) exceeds width of its type (1 bit)}}
  _Bool h : 5; // expected-error {{width of bit-field 'h' (5 bits) exceeds width of its type (1 bit)}}
};

struct b {unsigned x : 2;} x;
__typeof__(x.x+1) y;
int y;

struct {unsigned x : 2;} x2;
__typeof__((x.x+=1)+1) y;
__typeof__((0,x.x)+1) y;
__typeof__(x.x<<1) y;
int y;

struct PR8025 {
  double : 2; // expected-error{{anonymous bit-field has non-integral type 'double'}}
};

struct Test4 {
  unsigned bitX : 4;
  unsigned bitY : 4;
  unsigned var;
};
void test4(struct Test4 *t) {
  (void) sizeof(t->bitX); // expected-error {{invalid application of 'sizeof' to bit-field}}
  (void) sizeof((t->bitY)); // expected-error {{invalid application of 'sizeof' to bit-field}}
  (void) sizeof(t->bitX = 4); // not a bitfield designator in C
  (void) sizeof(t->bitX += 4); // not a bitfield designator in C
  (void) sizeof((void) 0, t->bitX); // not a bitfield designator in C
  (void) sizeof(t->var ? t->bitX : t->bitY); // not a bitfield designator in C
  (void) sizeof(t->var ? t->bitX : t->bitX); // not a bitfield designator in C
}

typedef unsigned Unsigned;
typedef signed Signed;

struct Test5 { unsigned n : 2; } t5;
// Bitfield is unsigned
struct Test5 sometest5 = {-1};
typedef __typeof__(+t5.n) Signed;  // ... but promotes to signed.

typedef __typeof__(t5.n + 0) Signed; // Arithmetic promotes.

typedef __typeof__(+(t5.n = 0)) Signed;  // FIXME: Assignment should not; the result
typedef __typeof__(+(t5.n += 0)) Signed; // is a non-bit-field lvalue of type unsigned.
typedef __typeof__(+(t5.n *= 0)) Signed;

typedef __typeof__(+(++t5.n)) Signed; // FIXME: Increment is equivalent to compound-assignment.
typedef __typeof__(+(--t5.n)) Signed; // This should not promote to signed.

typedef __typeof__(+(t5.n++)) Unsigned; // Post-increment is underspecified, but seems to
typedef __typeof__(+(t5.n--)) Unsigned; // also act like compound-assignment.

struct Test6 {
  : 0.0; // expected-error{{type name requires a specifier or qualifier}}
};

struct PR36157 {
  int n : 1 ? 1 : implicitly_declare_function(); // expected-warning {{invalid in C99}}
};
