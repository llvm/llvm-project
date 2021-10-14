// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -pedantic %s
// REQUIRES: LP64

struct foo { int a, b; };

static struct foo t = (struct foo){0,0};
static struct foo t1 = __builtin_choose_expr(0, (struct foo){0,0}, (struct foo){0,0});
static struct foo t2 = {0,0};
static struct foo t3 = t2; // expected-error {{initializer element is not a compile-time constant}}
static int *p = (int []){2,4};
static int x = (int){1};

static int *p2 = (int []){2,x}; // expected-error {{initializer element is not a compile-time constant}}
static long *p3 = (long []){2,"x"}; // expected-warning {{incompatible pointer to integer conversion initializing 'long' with an expression of type 'char [2]'}}

typedef struct { } cache_t; // expected-warning{{empty struct is a GNU extension}}
static cache_t clo_I1_cache = ((cache_t) { } ); // expected-warning{{use of GNU empty initializer extension}}

typedef struct Test {int a;int b;} Test;
static Test* ll = &(Test) {0,0};

extern void fooFunc(struct foo *pfoo);

int main(int argc, char **argv) {
 int *l = (int []){x, *p, *p2};
 fooFunc(&(struct foo){ 1, 2 });
}

struct Incomplete; // expected-note{{forward declaration of 'struct Incomplete'}}
struct Incomplete* I1 = &(struct Incomplete){1, 2, 3}; // expected-error {{variable has incomplete type}}
void IncompleteFunc(unsigned x) {
  struct Incomplete* I2 = (struct foo[x]){1, 2, 3}; // expected-error {{variable-sized object may not be initialized}}
  (void){1,2,3}; // expected-error {{variable has incomplete type}}
  (void(void)) { 0 }; // expected-error{{illegal initializer type 'void (void)'}}
}

// PR6080
int array[(sizeof(int[3]) == sizeof( (int[]) {0,1,2} )) ? 1 : -1];

// rdar://28949016 - Constant restriction should not apply to compound literals in blocks
int (^block)(int) = ^(int i) {
  int *array = (int[]) {i, i + 2, i + 4};
  return array[i];
};
