// RUN: %clang_cc1 %s -fblocks -verify -fsyntax-only

#if !__has_attribute(noescape)
#  error "missing noescape attribute"
#endif

int *global_var __attribute((noescape)); // expected-warning{{'noescape' attribute only applies to parameters}}

void foo(__attribute__((noescape)) int *int_ptr,
         __attribute__((noescape)) int (^block)(int),
         __attribute((noescape)) int integer) { // expected-warning{{'noescape' attribute ignored on parameter of non-pointer type 'int'}}
}
