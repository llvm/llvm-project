// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// The [[now_init]] marker (P4222R2 §6.2; its exact placement and spelling
// track an open committee question) is recognized regardless of -fprofiles.
// It applies to functions with at least one [[ref_to_uninit]] parameter --
// the parameters whose storage the callee promises to initialize; with no
// profile enforced it has no effect on those valid subjects. Other
// placements, and a function with no marked parameter (a vacuous promise),
// are rejected regardless of -fprofiles.

int g;

[[now_init]] void fill(int *p [[ref_to_uninit]]);
[[now_init]] void fill_ref(int &r [[ref_to_uninit]]);
void fill_after_name [[now_init]] (int *p [[ref_to_uninit]]);
[[now_init]] void fill_many(int n, int *p [[ref_to_uninit]], int *q);
[[now_init]] int *fill_and_return(int *p [[ref_to_uninit]]);

struct S {
  [[now_init]] void fill_member(int *p [[ref_to_uninit]]);
  [[now_init]] void fill_out_of_line(int *p [[ref_to_uninit]]);
};
void S::fill_out_of_line(int *p [[ref_to_uninit]]) {}

template <typename T>
[[now_init]] void fill_template(T *p [[ref_to_uninit]]);

// A dependent parameter's marker is attached to the pattern unvalidated, so
// the vacuity check accepts the template; if instantiation drops the marker
// (T deduced such that T p is not a pointer/reference), the inherited
// [[now_init]] is inert rather than re-diagnosed, like the dropped marker.
template <typename T>
[[now_init]] void fill_dependent(T p [[ref_to_uninit]]) {}
template void fill_dependent<int *>(int *);
template void fill_dependent<int>(int);

int bad_var [[now_init]]; // expected-error {{'now_init' attribute only applies to functions}} \
                          // no-profiles-error {{'now_init' attribute only applies to functions}}

struct BadField {
  int m [[now_init]]; // expected-error {{'now_init' attribute only applies to functions}} \
                      // no-profiles-error {{'now_init' attribute only applies to functions}}
};

[[now_init]] void bad_no_params(); // expected-error {{'now_init' attribute requires at least one parameter marked '[[ref_to_uninit]]'}} \
                                   // no-profiles-error {{'now_init' attribute requires at least one parameter marked '[[ref_to_uninit]]'}}

[[now_init]] void bad_unmarked_params(int *p, int &r); // expected-error {{'now_init' attribute requires at least one parameter marked '[[ref_to_uninit]]'}} \
                                                       // no-profiles-error {{'now_init' attribute requires at least one parameter marked '[[ref_to_uninit]]'}}

// The [[ref_to_uninit]] on a non-pointer parameter is itself rejected and
// dropped, leaving the function with no marked parameter: the vacuity error
// fires alongside.
[[now_init]] void bad_dropped_marker(int v [[ref_to_uninit]]); // expected-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                                               // expected-error {{'now_init' attribute requires at least one parameter marked '[[ref_to_uninit]]'}} \
                                                               // no-profiles-error {{'ref_to_uninit' attribute only applies to pointers, references, and functions returning them}} \
                                                               // no-profiles-error {{'now_init' attribute requires at least one parameter marked '[[ref_to_uninit]]'}}

// Inert without an enforced profile: a valid placement changes nothing about
// how either run type-checks these calls.
void use() {
  int x = 0;
  fill(&x);
  fill_ref(x);
}
