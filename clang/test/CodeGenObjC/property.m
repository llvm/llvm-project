// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// PR13820
// REQUIRES: LP64

// TODO: actually test most of this instead of just emitting it

int printf(const char *, ...);

@interface Root
-(id) alloc;
-(id) init;
@end

@interface A : Root {
  int x;
  int y, ro, z;
  id ob0, ob1, ob2, ob3, ob4;
}
@property int x;
@property int y;
@property int z;
@property(readonly) int ro;
@property(assign) id ob0;
@property(retain) id ob1;
@property(copy) id ob2;
@property(retain, nonatomic) id ob3;
@property(copy, nonatomic) id ob4;
@end

@implementation A
@dynamic x;
@synthesize y;
@synthesize z = z;
@synthesize ro;
@synthesize ob0;
@synthesize ob1;
@synthesize ob2;
@synthesize ob3;
@synthesize ob4;
-(int) y {
  return x + 1;
}
-(void) setZ: (int) arg {
  x = arg - 1;
}
@end

@interface A (Cat)
@property int dyn;
@end

@implementation A (Cat)
-(int) dyn {
  return 10;
}
@end

// Test that compound operations only compute the base once.
// CHECK: define void @test2
A *test2_helper(void);
void test2() {
  // CHECK:      [[BASE:%.*]] = call [[A:%.*]]* @test2_helper()
  // CHECK-NEXT: [[SEL:%.*]] = load i8**
  // CHECK-NEXT: [[BASETMP:%.*]] = bitcast [[A]]* [[BASE]] to i8*
  // CHECK-NEXT: [[LD:%.*]] = call i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* [[BASETMP]], i8* [[SEL]])
  // CHECK-NEXT: [[ADD:%.*]] = add nsw i32 [[LD]], 1
  // CHECK-NEXT: [[SEL:%.*]] = load i8**
  // CHECK-NEXT: [[BASETMP:%.*]] = bitcast [[A]]* [[BASE]] to i8*
  // CHECK-NEXT: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i32)*)(i8* [[BASETMP]], i8* [[SEL]], i32 [[ADD]])
  test2_helper().dyn++;

  // CHECK:      [[BASE:%.*]] = call [[A]]* @test2_helper()
  // CHECK-NEXT: [[SEL:%.*]] = load i8**
  // CHECK-NEXT: [[BASETMP:%.*]] = bitcast [[A]]* [[BASE]] to i8*
  // CHECK-NEXT: [[LD:%.*]] = call i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* [[BASETMP]], i8* [[SEL]])
  // CHECK-NEXT: [[ADD:%.*]] = mul nsw i32 [[LD]], 10
  // CHECK-NEXT: [[SEL:%.*]] = load i8**
  // CHECK-NEXT: [[BASETMP:%.*]] = bitcast [[A]]* [[BASE]] to i8*
  // CHECK-NEXT: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i32)*)(i8* [[BASETMP]], i8* [[SEL]], i32 [[ADD]])
  test2_helper().dyn *= 10;
}

// Test aggregate initialization from property reads.
// Not crashing is good enough for the property-specific test.
struct test3_struct { int x,y,z; };
struct test3_nested { struct test3_struct t; };
@interface test3_object
@property struct test3_struct s;
@end
void test3(test3_object *p) {
  struct test3_struct array[1] = { p.s };
  struct test3_nested agg = { p.s };
}

// PR8742
@interface Test4  {}
@property float f;
@end
// CHECK: define void @test4
void test4(Test4 *t) {
  extern int test4_printf(const char *, ...);
  // CHECK: [[TMP:%.*]] = call float {{.*}} @objc_msgSend
  // CHECK-NEXT: [[EXT:%.*]] = fpext float [[TMP]] to double
  // CHECK-NEXT: call i32 (i8*, ...)* @test4_printf(i8* {{.*}}, double [[EXT]])
  // CHECK-NEXT: ret void
  test4_printf("%.2f", t.f);
}

@interface Test5 {
  unsigned _x : 5;
}
@property unsigned x;
@end
@implementation Test5
@synthesize x = _x;
@end

// rdar://problem/10410531
@interface Test6
@property void (*prop)(void);
@end

void test6_func(void);
void test6(Test6 *a) {
  a.prop = test6_func;
}

// rdar://problem/10507455
@interface Test7
@property unsigned char x;
@end
void test7(Test7 *t) {
  t.x &= 2;
  t.x |= 5;
  t.x ^= 8;
}
// CHECK:    define void @test7([[TEST7:%.*]]*
// CHECK:      [[T:%.*]] = alloca [[TEST7]]*,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load [[TEST7]]** [[T]], align
// CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST7]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call zeroext i8 bitcast
// CHECK-NEXT: [[T3:%.*]] = zext i8 [[T2]] to i32
// CHECK-NEXT: [[T4:%.*]] = and i32 [[T3]], 2
// CHECK-NEXT: [[T5:%.*]] = trunc i32 [[T4]] to i8
// CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T6:%.*]] = bitcast [[TEST7]]* [[T0]] to i8*
// CHECK-NEXT: call void bitcast
// CHECK-NEXT: [[T0:%.*]] = load [[TEST7]]** [[T]], align
// CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST7]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call zeroext i8 bitcast
// CHECK-NEXT: [[T3:%.*]] = zext i8 [[T2]] to i32
// CHECK-NEXT: [[T4:%.*]] = or i32 [[T3]], 5
// CHECK-NEXT: [[T5:%.*]] = trunc i32 [[T4]] to i8
// CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T6:%.*]] = bitcast [[TEST7]]* [[T0]] to i8*
// CHECK-NEXT: call void bitcast
// CHECK-NEXT: [[T0:%.*]] = load [[TEST7]]** [[T]], align
// CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST7]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call zeroext i8 bitcast
// CHECK-NEXT: [[T3:%.*]] = zext i8 [[T2]] to i32
// CHECK-NEXT: [[T4:%.*]] = xor i32 [[T3]], 8
// CHECK-NEXT: [[T5:%.*]] = trunc i32 [[T4]] to i8
// CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T6:%.*]] = bitcast [[TEST7]]* [[T0]] to i8*
// CHECK-NEXT: call void bitcast
// CHECK-NEXT: ret void
