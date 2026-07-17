// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// The [[now_uninit]] marker -- the mirror of [[now_init]], supplying the
// recording P4222R2 §4.4 notes is missing for destroy_at -- is recognized
// regardless of -fprofiles. It applies to functions with at least one
// parameter of pointer or reference type -- the parameters whose storage the
// callee destroys; with no profile enforced it has no effect on those valid
// subjects. Other placements, and a function with no pointer or reference
// parameter (a vacuous claim), are rejected regardless of -fprofiles.

[[now_uninit]] void wipe(int *p);
[[now_uninit]] void wipe_ref(int &r);
[[now_uninit]] void wipe_void(void *p);
void wipe_after_name [[now_uninit]] (int *p);
[[now_uninit]] void wipe_many(int n, int *p, bool b);

// Carrying both attributes is allowed: a reinitializer both ends and starts
// a lifetime for its argument's storage.
[[now_init]] [[now_uninit]] void reinit(int *p [[ref_to_uninit]]);

struct S {
  [[now_uninit]] void wipe_member(int *p);
  [[now_uninit]] void wipe_out_of_line(int *p);
};
void S::wipe_out_of_line(int *p) {}

template <typename T>
[[now_uninit]] void wipe_template(T *p) {}

// A dependent parameter type may instantiate to a pointer or reference, so
// the vacuity check accepts the template; if it does not, the inherited
// attribute goes inert rather than re-diagnosed (the withdrawal arms only
// key on pointer/reference bindings).
template <typename T>
[[now_uninit]] void wipe_dependent(T v) {}
template void wipe_dependent<int *>(int *);
template void wipe_dependent<int>(int);

int bad_var [[now_uninit]]; // expected-error {{'now_uninit' attribute only applies to functions}} \
                            // no-profiles-error {{'now_uninit' attribute only applies to functions}}

struct BadField {
  int m [[now_uninit]]; // expected-error {{'now_uninit' attribute only applies to functions}} \
                        // no-profiles-error {{'now_uninit' attribute only applies to functions}}
};

[[now_uninit]] void bad_no_params(); // expected-error {{'now_uninit' attribute requires at least one parameter of pointer or reference type}} \
                                     // no-profiles-error {{'now_uninit' attribute requires at least one parameter of pointer or reference type}}

[[now_uninit]] void bad_value_params(int v, bool b); // expected-error {{'now_uninit' attribute requires at least one parameter of pointer or reference type}} \
                                                     // no-profiles-error {{'now_uninit' attribute requires at least one parameter of pointer or reference type}}

// A function pointer denotes a function, never destroyable storage
// (mirroring [[ref_to_uninit]]'s subject rule), so it does not satisfy the
// vacuity check.
[[now_uninit]] void bad_fn_ptr_param(void (*fp)()); // expected-error {{'now_uninit' attribute requires at least one parameter of pointer or reference type}} \
                                                    // no-profiles-error {{'now_uninit' attribute requires at least one parameter of pointer or reference type}}

// Inert without an enforced profile: a valid placement changes nothing about
// how either run type-checks these calls.
void use() {
  int x = 0;
  wipe(&x);
  wipe_ref(x);
  wipe_template<int>(&x);
}
