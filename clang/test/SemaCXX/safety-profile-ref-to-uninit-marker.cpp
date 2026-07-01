// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// The [[ref_to_uninit]] marker is recognized regardless of -fprofiles. It
// applies to pointers, references, and pointer/reference-returning functions;
// with no profile enforced it has no effect on those valid subjects. Other
// placements are rejected regardless of -fprofiles.

int g;

int *gp [[ref_to_uninit]] = &g;
int &gr [[ref_to_uninit]] = g;
void *gvp [[ref_to_uninit]] = &g;
[[ref_to_uninit]] int *gp_prefix = &g;

[[ref_to_uninit]] int *allocate(int n);
[[ref_to_uninit]] int &bind_ret();
void fill(int *p [[ref_to_uninit]]);
void bind(int &r [[ref_to_uninit]]);

struct S {
  int *m [[ref_to_uninit]];
};

int bad_scalar [[ref_to_uninit]]; // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                  // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}

[[ref_to_uninit]] int bad_array[3]; // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                    // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}

[[ref_to_uninit]] void bad_return(); // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                     // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}

struct BadMember {
  int m [[ref_to_uninit]]; // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                           // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}
};

// A function pointer or reference (or a function returning one) denotes a
// function, never uninitialized memory, so the marker can never be satisfied
// and is rejected -- like a pointer-to-member, which is not a pointer type.
void some_fn();

void (*bad_fn_ptr [[ref_to_uninit]])(); // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                        // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}

void (&bad_fn_ref [[ref_to_uninit]])() = some_fn; // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                                  // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}

[[ref_to_uninit]] void (*bad_ret_fn_ptr())(); // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                              // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}

struct C { void mf(); };
void (C::*bad_mem_fn_ptr [[ref_to_uninit]])(); // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                               // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}

struct BadFnPtrMember {
  void (*m [[ref_to_uninit]])(); // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                 // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}}
};
