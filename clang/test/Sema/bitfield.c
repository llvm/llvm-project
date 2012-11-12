// RUN: %clang_cc1 %s -fsyntax-only -verify 
enum e0; // expected-note{{forward declaration of 'enum e0'}}

struct a {
  int a : -1; // expected-error{{bit-field 'a' has negative width}}

  // rdar://6081627
  int b : 33; // expected-error{{size of bit-field 'b' (33 bits) exceeds size of its type (32 bits)}}

  int c : (1 + 0.25); // expected-error{{expression is not an integer constant expression}}
  int d : (int)(1 + 0.25); 

  // rdar://6138816
  int e : 0;  // expected-error {{bit-field 'e' has zero width}}

  float xx : 4;  // expected-error {{bit-field 'xx' has non-integral type}}

  // PR3607
  enum e0 f : 1; // expected-error {{field has incomplete type 'enum e0'}}
  
  int g : (_Bool)1;
  
  // PR4017  
  char : 10;      // expected-error {{size of anonymous bit-field (10 bits) exceeds size of its type (8 bits)}}
  unsigned : -2;  // expected-error {{anonymous bit-field has negative width (-2)}}
  float : 12;     // expected-error {{anonymous bit-field has non-integral type 'float'}}
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
