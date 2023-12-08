@import redecl_merge_left;

@class C4;
@class C4;
@protocol P4;
@protocol P4;
@protocol P4;
@import redecl_merge_right;

@class B;

@class A;

@protocol P1;

struct S1;
struct S3;

void refers_to_C4(C4*);

@interface UnrelatedToDeclaredThenLoaded
- declaredThenLoadedMethod;
@end

@class DeclaredThenLoaded;
