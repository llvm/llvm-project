// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// CHECK: @"_OBJC_$_PROTOCOL_METHOD_TYPES_P1" = internal global
// CHECK: @[[PROTO_P1:"_OBJC_PROTOCOL_\$_P1"]] = weak hidden
// CHECK: @[[LABEL_PROTO_P1:"_OBJC_LABEL_PROTOCOL_\$_P1"]] = weak hidden global %{{.*}}* @[[PROTO_P1]]
// CHECK: @[[PROTO_P2:"_OBJC_PROTOCOL_\$_P2"]] = weak hidden
// CHECK: @[[LABEL_PROTO_P2:"_OBJC_LABEL_PROTOCOL_\$_P2"]] = weak hidden global %{{.*}}* @[[PROTO_P2]]
// CHECK: @"_OBJC_$_PROTOCOL_REFS_P3" = internal global { i64, [3 x %{{.*}}] } { i64 2, [3 x %{{.*}}*] [%{{.*}}* @[[PROTO_P1]], %{{.*}}* @[[PROTO_P2]], %{{.*}}* null] }
// CHECK: @[[PROTO_P3:"_OBJC_PROTOCOL_\$_P3"]] = weak hidden
// CHECK: @[[LABEL_PROTO_P3:"_OBJC_LABEL_PROTOCOL_\$_P3"]] = weak hidden global %{{.*}}* @[[PROTO_P3]]
// CHECK: "_OBJC_PROTOCOL_REFERENCE_$_P3" = weak hidden global %{{.*}}* bitcast (%{{.*}}* @[[PROTO_P3]] to %{{.*}}*)
// CHECK: @[[PROTO_P0:"_OBJC_PROTOCOL_\$_P0"]] = weak hidden
// CHECK: @[[LABEL_PROTO_P0:"_OBJC_LABEL_PROTOCOL_\$_P0"]] = weak hidden global %{{.*}}* @[[PROTO_P0]]
// CHECK: "_OBJC_PROTOCOL_REFERENCE_$_P0" = weak hidden global %0* bitcast (%{{.*}}* @[[PROTO_P0]] to %{{.*}}*)
// CHECK: "_OBJC_PROTOCOL_REFERENCE_$_P1" = weak hidden global %0* bitcast (%{{.*}}* @[[PROTO_P1]] to %{{.*}}*)
// CHECK: "_OBJC_PROTOCOL_REFERENCE_$_P2" = weak hidden global %0* bitcast (%{{.*}}* @[[PROTO_P2]] to %{{.*}}*)

void p(const char*, ...);

@interface Root
+(int) maxValue;
-(int) conformsTo: (id) x;
@end

@protocol P0
@end

@protocol P1
+(void) classMethodReq0;
-(void) methodReq0;
@optional
+(void) classMethodOpt1;
-(void) methodOpt1;
@required
+(void) classMethodReq2;
-(void) methodReq2;
@end

@protocol P2
//@property(readwrite) int x;
@end

@protocol P3<P1, P2>
-(id <P1>) print0;
-(void) print1;
@end

void foo(const id a) {
  void *p = @protocol(P3);
}

int main(void) {
  Protocol *P0 = @protocol(P0);
  Protocol *P1 = @protocol(P1);
  Protocol *P2 = @protocol(P2);
  Protocol *P3 = @protocol(P3);

#define Pbool(X) p(#X ": %s\n", X ? "yes" : "no");
  Pbool([P0 conformsTo: P1]);
  Pbool([P1 conformsTo: P0]);
  Pbool([P1 conformsTo: P2]);
  Pbool([P2 conformsTo: P1]);
  Pbool([P3 conformsTo: P1]);
  Pbool([P1 conformsTo: P3]);

  return 0;
}

// rdar://problem/7992749
typedef Root<P1> P1Object;
int test10(void) {
  return [P1Object maxValue];
}
