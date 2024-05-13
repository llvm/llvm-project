// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}

// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -include %s %s

// With PCH
// RUN: %clang_cc1 %s -emit-pch -o %t
// RUN: %clang_cc1 -emit-llvm-only -verify %s -include-pch %t -debug-info-kind=limited

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
//===----------------------------------------------------------------------===//
// Header

typedef char BOOL;

@interface NSString
+ (BOOL)meth;
@end

static NSString * const cake = @"cake";

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

@interface Foo {
  BOOL ivar;
}
@end

//===----------------------------------------------------------------------===//
#endif
