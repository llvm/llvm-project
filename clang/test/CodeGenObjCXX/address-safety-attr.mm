// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix=ASAN %s

// REQUIRES: more-investigation

@interface MyClass
+ (int) addressSafety:(int*)a;
@end

@implementation MyClass

// WITHOUT:  +[MyClass load]{{.*}}#[[ATTR0:[0-9]+]]
// ASAN: +[MyClass load]{{.*}}#[[ATTR1:[0-9]+]]
+(void) load { }

// WITHOUT:  +[MyClass addressSafety:]{{.*}}#[[ATTR0]]
// ASAN:  +[MyClass addressSafety:]{{.*}}#[[ATTR2:[0-9]+]]
+ (int) addressSafety:(int*)a { return *a; }

@end

// ASAN : attributes #[[ATTR1]] = {{.*}}sanitized_padded_global
// ASAN : attributes #[[ATTR2]] = {{.*}}sanitize_address
// WITHOUT-NOT: attributes #[[ATTR0]] = {{.*}}sanitize_address
