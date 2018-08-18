// RUN: %clang_cc1 -emit-llvm -triple i686-apple-darwin8 -fobjc-runtime=macosx-fragile-10.5 -o %t %s

// No object generated
// RUN: not grep OBJC_PROTOCOL_P0 %t
@protocol P0;

// No object generated
// RUN: not grep OBJC_PROTOCOL_P1 %t
@protocol P1 -im1; @end

// Definition triggered by protocol reference.
// RUN: grep OBJC_PROTOCOL_P2 %t | count 3
// RUN: grep OBJC_PROTOCOL_INSTANCE_METHODS_P2 %t | count 3
@protocol P2 -im1; @end
void f0() { id x = @protocol(P2); }

// Forward definition triggered by protocol reference.
// RUN: grep OBJC_PROTOCOL_P3 %t | count 3
// RUN: not grep OBJC_PROTOCOL_INSTANCE_METHODS_P3 %t
@protocol P3;
@interface UserP3<P3>
@end
@implementation UserP3
@end

// Definition triggered by class reference.
// RUN: grep OBJC_PROTOCOL_P4 %t | count 3
// RUN: grep OBJC_PROTOCOL_INSTANCE_METHODS_P4 %t | count 3
@protocol P4 -im1; @end
@interface I0<P4> @end
@implementation I0 -im1 { return 0; }; @end

// Definition following forward reference.
// RUN: grep OBJC_PROTOCOL_P5 %t | count 3
// RUN: grep OBJC_PROTOCOL_INSTANCE_METHODS_P5 %t | count 3
@protocol P5;
@interface UserP5<P5> // This generates a forward
                      // reference, which has to be
                      // updated on the next line.
@end
@protocol P5 -im1; @end
@implementation UserP5

- im1 { }

@end

// Protocol reference following definition.
// RUN: grep OBJC_PROTOCOL_P6 %t | count 4
// RUN: grep OBJC_PROTOCOL_INSTANCE_METHODS_P6 %t | count 3
@protocol P6 -im1; @end
@interface I1<P6> @end
@implementation I1 -im1 { return 0; }; @end
void f3() { id x = @protocol(P6); }

