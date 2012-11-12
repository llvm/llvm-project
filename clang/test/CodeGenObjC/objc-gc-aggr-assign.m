// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o - %s | FileCheck -check-prefix C %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o - %s | FileCheck -check-prefix CP %s

static int count;

typedef struct S {
   int ii;
} SS;

struct type_s {
   SS may_recurse;
   id id_val;
};

@interface NamedObject
{
  struct type_s type_s_ivar;
}
- (void) setSome : (struct type_s) arg;
- (struct type_s) getSome;
@property(assign) struct type_s aggre_prop;
@end

@implementation NamedObject 
- (void) setSome : (struct type_s) arg
  {
     type_s_ivar = arg;
  }
- (struct type_s) getSome 
  {
    return type_s_ivar;
  }
@synthesize aggre_prop = type_s_ivar;
@end

struct type_s some = {{1234}, (id)0};

struct type_s get(void)
{
  return some;
}

void f(const struct type_s *in, struct type_s *out) {
  *out = *in;
}

#ifdef __cplusplus
struct Derived : type_s { };

void foo(Derived* src, Derived* dest) {
        *dest = *src;
}
#endif

// CHECK-C: call i8* @objc_memmove_collectable
// CHECK-C: call i8* @objc_memmove_collectable
// CHECK-C: call i8* @objc_memmove_collectable

// CHECK-CP: call i8* @objc_memmove_collectable
// CHECK-CP: call i8* @objc_memmove_collectable
// CHECK-CP: call i8* @objc_memmove_collectable
// CHECK-CP: call i8* @objc_memmove_collectable
