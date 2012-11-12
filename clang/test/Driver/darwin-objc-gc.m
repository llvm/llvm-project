// Check that we warn, but accept, -fobjc-gc for iPhone OS.

// RUN: %clang -target i386-apple-darwin9 -miphoneos-version-min=3.0 -fobjc-gc -flto -S -o %t %s 2> %t.err
// RUN: FileCheck --check-prefix=IPHONE_OBJC_GC_LL %s < %t 
// RUN: FileCheck --check-prefix=IPHONE_OBJC_GC_STDERR %s < %t.err

// IPHONE_OBJC_GC_LL: define void @f0
// IPHONE_OBJC_GC_LL-NOT: objc_assign_ivar
// IPHONE_OBJC_GC_LL: }

// IPHONE_OBJC_GC_STDERR: warning: Objective-C garbage collection is not supported on this platform, ignoring '-fobjc-gc'

@interface A {
@public
 id x;
}
@end

void f0(A *a, id x) { a->x = x; }
