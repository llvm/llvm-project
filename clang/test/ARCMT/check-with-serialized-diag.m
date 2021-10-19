
@protocol NSObject
- (id)retain;
- (unsigned)retainCount;
- (oneway void)release;
- (id)autorelease;
@end

@interface NSObject <NSObject> {}
- (id)init;

+ (id)new;
+ (id)alloc;
- (void)dealloc;

- (void)finalize;

- (id)copy;
- (id)mutableCopy;
@end

@interface A : NSObject
@end

struct UnsafeS {
  A *__unsafe_unretained unsafeObj;
};

id global_foo;

void test1(A *a, struct UnsafeS *unsafeS) {
  [unsafeS->unsafeObj retain];
  id foo = [unsafeS->unsafeObj retain]; // no warning.
  [global_foo retain];
  [a retainCount];
}

// RUN: not %clang_cc1 -arcmt-action=check -triple x86_64-apple-darwin10 %s -serialize-diagnostic-file %t.diag
// RUN: c-index-test -read-diagnostics %t.diag > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// CHECK: {{.*}}check-with-serialized-diag.m:32:4: error: [rewriter] it is not safe to remove 'retain' message on an __unsafe_unretained type
// CHECK-NEXT: Number FIXITs = 0
// CHECK-NEXT: {{.*}}check-with-serialized-diag.m:34:4: error: [rewriter] it is not safe to remove 'retain' message on a global variable
// CHECK-NEXT: Number FIXITs = 0
// CHECK-NEXT: {{.*}}check-with-serialized-diag.m:32:23: error: ARC forbids explicit message send of 'retain'
// CHECK-NEXT: Range: {{.*}}check-with-serialized-diag.m:32:4 {{.*}}check-with-serialized-diag.m:32:22
// CHECK-NEXT: Number FIXITs = 0
// CHECK-NEXT: {{.*}}check-with-serialized-diag.m:34:15: error: ARC forbids explicit message send of 'retain'
// CHECK-NEXT: Range: {{.*}}check-with-serialized-diag.m:34:4 {{.*}}check-with-serialized-diag.m:34:14
// CHECK-NEXT: Number FIXITs = 0
// CHECK-NEXT: {{.*}}check-with-serialized-diag.m:35:6: error: ARC forbids explicit message send of 'retainCount'
// CHECK-NEXT: Range: {{.*}}check-with-serialized-diag.m:35:4 {{.*}}check-with-serialized-diag.m:35:5
// CHECK-NEXT: Number FIXITs = 0

