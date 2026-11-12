// RUN: %clang_cc1 -ast-print -x objective-c++ -fms-extensions %s -o - | FileCheck %s

// CHECK: using A = __kindof id (*)[1];
using A = __kindof id (*)[1];

// CHECK: using B = int ** __ptr32 *[3];
using B = int ** __ptr32 *[3];

// FIXME: This is the wrong spelling for the attribute.
// FIXME: Too many parens here!
// CHECK: using C = int ((*))() __attribute__((cdecl));
using C = int (*)() [[gnu::cdecl]];

// CHECK: int fun_asm() asm("test");
int fun_asm() asm("test");
// CHECK: int var_asm asm("test");
int var_asm asm("test");


@interface NSString
@end

extern NSString *const MyErrorDomain;
// CHECK: enum __attribute__((ns_error_domain(MyErrorDomain))) MyErrorEnum {
enum __attribute__((ns_error_domain(MyErrorDomain))) MyErrorEnum {
  MyErrFirst,
  MyErrSecond,
};

// CHECK: int *fun_returns() __attribute__((ownership_returns(fun_returns)));
int *fun_returns() __attribute__((ownership_returns(fun_returns)));

// CHECK: void fun_holds(int *a) __attribute__((ownership_holds(fun_holds, 1)));
void fun_holds(int *a) __attribute__((ownership_holds(fun_holds, 1)));

// CHECK: int fun_var_unused() {
// CHECK-NEXT: int x __attribute__((unused)) = 0;
// CHECK-NEXT: return x;
// CHECK-NEXT: }
int fun_var_unused() { int x __attribute__((unused)) = 0; return x; }
