@import redecl_merge_top;

@interface Super
@end

@interface A : Super
- (Super*)init;
@end

@class B;

@protocol P1
- (void)protoMethod1;
@end

@protocol P1;

@protocol P2;

@protocol P2;

@protocol P2;

struct S1;
struct S2;

void consume_S1(struct S1*);
struct S2 *produce_S2(void);

// Test declarations in different modules with no common initial
// declaration.
@class C;
C *get_a_C(void);
@class C2;
C2 *get_a_C2(void);
@class C3;
C3 *get_a_C3(void);

@class C4;
@class C4;
@class C4;
@class C4;
C4 *get_a_C4(void);

@class Explicit;

int *explicit_func(void);

struct explicit_struct;

@protocol P4, P3;
@protocol P3;
@protocol P3;
@protocol P3;

struct S3;
struct S4;

void consume_S3(struct S3*);
struct S4 *produce_S4(void);

typedef int T1;
typedef double T2;

int func0(int);
int func1(int);
int func1(int);
int func1(int x) { return x; }
int func1(int);
static int func2(int);




// Spacing matters!
extern int var1;
extern int var2;

static double var3;

int ONE;
@import redecl_merge_top.Explicit;
const int one = ONE;

@interface ClassWithDef 
- (void)method;
@end

void eventually_noreturn(void) __attribute__((noreturn));
void eventually_noreturn2(void) __attribute__((noreturn));
