// RUN: %clang_cc1 -triple i686-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: FileCheck < %t %s
//
// CHECK: @"\01L_OBJC_METH_VAR_TYPE_34" = internal global [16 x i8] c"v12@0:4[3[4@]]8\00"

@class Int1;

struct Innermost {
  unsigned char a, b;
};

@interface Int1 {
  signed char a, b;
  struct Innermost *innermost;
}
@end

@implementation Int1
@end

@interface Base
{
    struct objc_class *isa;
    int full;
    int full2: 32;
    int _refs: 8;
    int field2: 3;
    unsigned f3: 8;
    short cc;
    unsigned g: 16;
    int r2: 8;
    int r3: 8;
    int r4: 2;
    int r5: 8;
    char c;
}
@end

@interface Derived: Base
{
    char d;
    int _field3: 6;
}
@end

@implementation Base
@end

@implementation Derived
@end

@interface B1 
{
    struct objc_class *isa;
    Int1 *sBase;
    char c;
}
@end

@implementation B1
@end

@interface Test 
{
	int ivar;
         __attribute__((objc_gc(weak))) SEL selector;
}
-(void) test3: (Test*  [3] [4])b ; 
- (SEL**) meth : (SEL) arg : (SEL*****) arg1 : (SEL*)arg2 : (SEL**) arg3;
@end

@implementation Test
-(void) test3: (Test* [3] [4])b {}
- (SEL**) meth : (SEL) arg : (SEL*****) arg1 : (SEL*)arg2 : (SEL**) arg3 {}
@end

struct S { int iS; };

@interface Object
{
 Class isa;
}
@end
typedef Object MyObj;

int main()
{
	const char *en = @encode(Derived);
	const char *eb = @encode(B1);
        const char *es = @encode(const struct S *);
        const char *ec = @encode(const struct S);
        const char *ee = @encode(MyObj *const);
}

// CHECK: @g0 = constant [15 x i8] c"{Innermost=CC}\00"
const char g0[] = @encode(struct Innermost);

// CHECK: @g1 = constant [38 x i8] c"{Derived=#ib32b8b3b8sb16b8b8b2b8ccb6}\00"
const char g1[] = @encode(Derived);

// CHECK: @g2 = constant [9 x i8] c"{B1=#@c}\00"
const char g2[] = @encode(B1);

// CHECK: @g3 = constant [8 x i8] c"r^{S=i}\00"
const char g3[] = @encode(const struct S *);

// CHECK: @g4 = constant [6 x i8] c"{S=i}\00"
const char g4[] = @encode(const struct S);

// CHECK: @g5 = constant [12 x i8] c"^{Object=#}\00"
const char g5[] = @encode(MyObj * const);

////

enum Enum1X { one, two, three, four };

@interface Base1X {
  unsigned a: 2;
  int b: 3;
  enum Enum1X c: 4;
  unsigned d: 5;
} 
@end

@interface Derived1X: Base1X {
  signed e: 5;
  int f: 4;
  enum Enum1X g: 3;
} 
@end

@implementation Base1X @end

@implementation Derived1X @end

// CHECK: @g6 = constant [18 x i8] c"{Base1X=b2b3b4b5}\00"
const char g6[] = @encode(Base1X);

// CHECK: @g7 = constant [27 x i8] c"{Derived1X=b2b3b4b5b5b4b3}\00"
const char g7[] = @encode(Derived1X);

// CHECK: @g8 = constant [7 x i8] c"{s8=D}\00"
struct s8 {
  long double x;
};
const char g8[] = @encode(struct s8);

// CHECK: @g9 = constant [11 x i8] c"{S9=i[0i]}\00"
struct S9 {
  int x;
  int flex[];
};
const char g9[] = @encode(struct S9);

struct f
{
  int i;
  struct{} g[4];
  int tt;
};

// CHECK: @g10 = constant [14 x i8] c"{f=i[0{?=}]i}\00"
const char g10[] = @encode(struct f);

// rdar://9622422
// CHECK: @g11 = constant [2 x i8] c"v\00"
const char g11[] = @encode(void);
