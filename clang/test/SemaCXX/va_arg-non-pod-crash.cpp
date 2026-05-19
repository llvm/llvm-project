// RUN: %clang_cc1 -emit-llvm-only -verify -Wno-everything %s

struct NonPODType {
  NonPODType(const NonPODType&);
  NonPODType& operator=(const NonPODType&);
  NonPODType(NonPODType&&);
  NonPODType& operator=(NonPODType&&);
};

void f(int first_arg, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, first_arg);
  NonPODType s = __builtin_va_arg(args, NonPODType); // expected-error {{second argument to 'va_arg' is of non-POD type 'NonPODType'}}
  __builtin_va_end(args);
}
