// RUN: %check_clang_tidy %s cppcoreguidelines-init-variables %t -- -- -fobjc-arc

@interface NSObject
@end

@interface NSNumber : NSObject
@end

@protocol NSFastEnumeration
@end

@interface NSArray<ObjectType> : NSObject <NSFastEnumeration>
@end

void testFastEnumeration(NSArray<NSNumber *> *array) {
  for (NSNumber *n in array) {
  }
}

// Regression test for https://github.com/llvm/llvm-project/issues/173435.
// CHECK-MESSAGES-NOT: warning: variable 'n' is not initialized
