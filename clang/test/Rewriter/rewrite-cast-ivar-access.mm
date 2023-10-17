// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix CHECK-LP --input-file=%t-rw.cpp %s

@interface F {
  int supervar;
}
@end

@interface G : F {
@public
  int ivar;
}
@end

@implementation G
- (void)foo:(F *)arg {
        int q = arg->supervar;
        int v = ((G *)arg)->ivar;
}
@end

void objc_assign_strongCast(id);
void __CFAssignWithWriteBarrier(void **location, void *value) {
        objc_assign_strongCast((id)value);
}

@interface RealClass {
        @public
        int f;
}
@end

@implementation RealClass
@end

@interface Foo {
        id reserved;
}
@end

@implementation Foo
- (void)bar {
        ((RealClass*)reserved)->f = 99;
}
@end

// CHECK-LP: ((struct G_IMPL *)arg)->ivar

// CHECK-LP: objc_assign_strongCast((id)value)

// CHECK-LP: ((struct RealClass_IMPL *)((RealClass *)((struct Foo_IMPL *)self)->reserved))->f
