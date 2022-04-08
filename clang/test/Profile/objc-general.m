// Test instrumentation of general constructs in objective C.

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.9 -main-file-name objc-general.m %s -o - -emit-llvm -fblocks -fprofile-instrument=clang | FileCheck -check-prefix=PGOGEN %s

// RUN: llvm-profdata merge %S/Inputs/objc-general.proftext -o %t.profdata
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.9 -main-file-name objc-general.m %s -o - -emit-llvm -fblocks -fprofile-instrument-use-path=%t.profdata 2>&1 | FileCheck -check-prefix=PGOUSE %s

// PGOUSE-NOT: warning: profile data may be out of date

#ifdef HAVE_FOUNDATION

// Use this to build an instrumented version to regenerate the input file.
#import <Foundation/Foundation.h>

#else

// Minimal definitions to get this to compile without Foundation.h.

@protocol NSObject
@end

@interface NSObject <NSObject>
- (id)init;
+ (id)alloc;
@end

struct NSFastEnumerationState;
@interface NSArray : NSObject
- (unsigned long) countByEnumeratingWithState: (struct NSFastEnumerationState*) state
                  objects: (id*) buffer
                  count: (unsigned long) bufferSize;
+(NSArray*) arrayWithObjects: (id) first, ...;
@end;
#endif

// PGOGEN: @[[FRC:"__profc_objc_general.m_\+\[A foreach_\]"]] = private global [2 x i64] zeroinitializer
// PGOGEN: @[[BLC:"__profc_objc_general.m___13\+\[A foreach_\]_block_invoke"]] = private global [2 x i64] zeroinitializer
// PGOGEN: @[[MAC:__profc_main]] = private global [1 x i64] zeroinitializer

@interface A : NSObject
+ (void)foreach: (NSArray *)array;
@end

@implementation A
// PGOGEN: define {{.*}}+[A foreach:]
// PGOUSE: define {{.*}}+[A foreach:]
// PGOGEN: store {{.*}} @[[FRC]], i32 0, i32 0
+ (void)foreach: (NSArray *)array
{
  __block id result;
  // PGOGEN: store {{.*}} @[[FRC]], i32 0, i32 1
  // PGOUSE: br {{.*}} !prof ![[FR1:[0-9]+]]
  // PGOUSE: br {{.*}} !prof ![[FR2:[0-9]+]]
  for (id x in array) {
    // PGOGEN: define {{.*}}_block_invoke
    // PGOUSE: define {{.*}}_block_invoke
    // PGOGEN: store {{.*}} @[[BLC]], i32 0, i32 0
    ^{
      static int init = 0;
      // PGOGEN: store {{.*}} @[[BLC]], i32 0, i32 1
      // PGOUSE: br {{.*}} !prof ![[BL1:[0-9]+]]
      if (init)
        result = x;
      init = 1;
    }();
  }
}
@end

void nested_objc_for_ranges(NSArray *arr) {
  int x = 0;
  for (id a in arr)
    for (id b in arr)
      ++x;
}

void consecutive_objc_for_ranges(NSArray *arr) {
  int x = 0;
  for (id a in arr) {}
  for (id b in arr)
    ++x;
}

// PGOUSE-DAG: ![[FR1]] = !{!"branch_weights", i32 2, i32 3}
// PGOUSE-DAG: ![[FR2]] = !{!"branch_weights", i32 3, i32 2}
// PGOUSE-DAG: ![[BL1]] = !{!"branch_weights", i32 2, i32 2}

int main(int argc, const char *argv[]) {
  A *a = [[A alloc] init];
  NSArray *array = [NSArray arrayWithObjects: @"0", @"1", (void*)0];
  [A foreach: array];
  return 0;
}
