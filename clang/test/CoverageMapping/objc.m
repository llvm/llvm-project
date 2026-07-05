// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name objc.m -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -w %s | FileCheck %s

@interface A
- (void)bork:(int)msg;
@end

                      // CHECK: func
void func(A *a) {     // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+3]]:2 = #0
  if (a)              // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:8 = #0
    [a bork:  20  ];  // CHECK: File 0, [[@LINE]]:5 -> [[@LINE]]:20 = #1
}

@interface NSArray
+ (NSArray*) arrayWithObjects: (id) first, ...;
- (unsigned) count;
@end

                               // CHECK: func2
void func2(NSArray *array) {   // CHECK-NEXT: File 0, [[@LINE]]:28 -> {{[0-9]+}}:2 = #0
  int i = 0;                   // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:28 -> [[@LINE+1]]:29 = #1
  for (NSArray *x in array) {  // CHECK-NEXT: File 0, [[@LINE]]:29 -> [[@LINE+7]]:4 = #1
                               // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:10 = #1
    if (x) {                   // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = #2
      i = 1;
    } else {                   // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = (#1 - #2)
      i = -1;
    }
  }
  i = 0;
}
