@import redecl_merge_top;

@class A;

@class A;

@interface B
+ (B*) create_a_B;
@end

@class A;

@protocol P1;
@protocol P2
- (void)protoMethod2;
@end

struct S1;
struct S2 {
  int field;
};

struct S1 *produce_S1(void);
void consume_S2(struct S2*);

// Test declarations in different modules with no common initial
// declaration.
@class C;
void accept_a_C(C*);

@class C2;
void accept_a_C2(C2*);

@class C3;
void accept_a_C3(C3*);
@class C3;

@class C4;

@class Explicit;

int *explicit_func(void);

struct explicit_struct;

@protocol P3, P4;

@protocol P3;

struct S3;
struct S3;
struct S4 {
  int field;
};

struct S3 *produce_S3(void);
void consume_S4(struct S4*);

typedef int T1;
typedef float T2;

int func0(int);
int func1(int x) { return x; }
int func2(int);










// Spacing matters!
extern int var1;
extern float var2;

extern double var3;

// Make sure this doesn't introduce an ambiguity-creating 'id' at the
// top level.
typedef void funcptr_with_id(int id);

// A class that is declared in the 'bottom' module, then loaded from
// one of the modules it depends on.
@interface DeclaredThenLoaded
- declaredThenLoadedMethod;
@end

@class DeclaredThenLoaded;

void eventually_noreturn2(void);
