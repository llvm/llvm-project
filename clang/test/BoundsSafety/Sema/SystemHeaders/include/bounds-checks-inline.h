// Pretend this is a system header
#pragma clang system_header
#include <ptrcheck.h>

int* get_unspecified_ptr(void);
int* __unsafe_indexable get_unsafe_ptr(void);
// strict-note@+1 2{{passing argument to parameter 'ptr' here}}
void receive_cb(int* __counted_by(count) ptr, int count);

// TODO: The `incorrect-error`s and `common-incorrect-error`s should not be emitted.
// They are a result of a bug
// and we emit the diagnostic rather than crashing (rdar://139815437).

// FIXME: The `FIXME-common-incorrect-error`` should be emitted but aren't for
// some reason (rdar://140145190).

inline int* __counted_by(count) inline_header_func_get_unspecified_ptr(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers.
  // strict-error@+2{{returning 'int *' from a function with incompatible result type 'int *__single __counted_by(count)' (aka 'int *__single') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // incorrect-error@+1{{cannot extract the lower bound of 'int *' because it has no bounds specification}}
  return get_unspecified_ptr();
}

inline int* __counted_by(count) inline_header_func_get_unsafe_ptr(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers.
  // strict-error@+2{{returning 'int *__unsafe_indexable' from a function with incompatible result type 'int *__single __counted_by(count)' (aka 'int *__single') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // incorrect-error@+1{{cannot extract the lower bound of 'int *__unsafe_indexable' because it has no bounds specification}}
  return get_unsafe_ptr();
}

inline int* __counted_by(count) inline_header_ret_explicit_unspecified_cast_0(int count) {
  // Outside of system headers this implicit conversion **is allowed**
  return (int*)0;
}


inline int* __counted_by(count) inline_header_ret_explicit_unsafe_indexable_cast_0(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers.
  return (int* __unsafe_indexable)0;
}

inline int* __counted_by(count) inline_header_ret_0(int count) {
  // Outside of system headers this implicit conversion **is allowed**
  return 0;
}

inline int* __counted_by(count) inline_header_ret_void_star_unspecified_0(int count) {
  // Outside of system headers this implicit conversion **is allowed**
  return (void*)0;
}

inline int* __counted_by(count) inline_header_ret_void_star_unsafe_indexable_0(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers.
  return (void* __unsafe_indexable) 0;
}

inline void inline_header_call_receive_cb_cast_0(void) {
  receive_cb((int*)0, 0);
}

inline void inline_header_call_receive_cb_cast_unspecified_ptr(void) {
  // strict-error@+1{{passing 'int *' to parameter of incompatible type 'int *__single __counted_by(count)' (aka 'int *__single') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  receive_cb(get_unspecified_ptr(), 0);
}

inline void inline_header_call_receive_cb_cast_unsafe_ptr(void) {
  // strict-error@+1{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single __counted_by(count)' (aka 'int *__single') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  receive_cb(get_unsafe_ptr(), 0);
}

inline void inline_header_assign_local_cb_cast_0(void) {
  int count = 0;
  int* __counted_by(count) local = (int*)0;

  local = (int*)0;
  count = 0;
}


void side_effect(void);


inline void inline_header_init_local_cb_unspecified_ptr(void) {
  // common-error@+1{{local variable count must be declared right next to its dependent decl}}
  int count = 0;
  // strict-error@+3{{initializing 'int *__single __counted_by(count)' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // common-error@+2{{local variable local must be declared right next to its dependent decl}}
  // common-incorrect-error@+1{{cannot extract the lower bound of 'int *' because it has no bounds specification}}
  int* __counted_by(count) local = get_unspecified_ptr();
}

inline void inline_header_assign_local_cb_unspecified_ptr(void) {
  int count2 = 0;
  int* __counted_by(count2) local2;

  side_effect();

  // strict-error@+3{{assigning to 'int *__single __counted_by(count2)' (aka 'int *__single') from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // FIXME: The errors in the previous function stop this error from appearing for some reason.
  // FIXME-common-incorrect-error@+1{{cannot extract the lower bound of 'int *' because it has no bounds specification}}
  local2 = get_unspecified_ptr();
  // strict-error@+1{{assignment to 'count2' requires corresponding assignment to 'int *__single __counted_by(count2)' (aka 'int *__single') 'local2'; add self assignment 'local2 = local2' if the value has not changed}}
  count2 = 0;
}

inline void inline_header_init_local_cb_unsafe_ptr(void) {
  // common-error@+1{{local variable count must be declared right next to its dependent decl}}
  int count = 0;
  // strict-error@+3{{initializing 'int *__single __counted_by(count)' (aka 'int *__single') with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // common-error@+2{{local variable local must be declared right next to its dependent decl}}
  // common-incorrect-error@+1{{cannot extract the lower bound of 'int *__unsafe_indexable' because it has no bounds specification}}
  int* __counted_by(count) local = get_unsafe_ptr();
}

inline void inline_header_assign_local_cb_unsafe_ptr(void) {
  int count2 = 0;
  int* __counted_by(count2) local2;

  side_effect();

  // strict-error@+3{{assigning to 'int *__single __counted_by(count2)' (aka 'int *__single') from incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // FIXME: The errors in the previous functions stop this error from appearing for some reason.
  // FIXME-common-incorrect-error@+1{{cannot extract the lower bound of 'int *' because it has no bounds specification}}
  local2 = get_unsafe_ptr();
  // strict-error@+1{{assignment to 'count2' requires corresponding assignment to 'int *__single __counted_by(count2)' (aka 'int *__single') 'local2'; add self assignment 'local2 = local2' if the value has not changed}}
  count2 = 0;
}
struct simple_cb {
  int count;
  int* __counted_by(count) ptr;
};


inline void inline_header_compound_literal_unspecified_ptr(struct simple_cb s, int* unspecified_ptr) {
  // FIXME: This diagnostic isn't emitted when there are diagnostics from other functions.
  // strict-error@+2{{initializing 'int *__single __counted_by(count)' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // fixme-incorrect-error@+1{{cannot extract the lower bound of 'int *' because it has no bounds specification}}
  s = (struct simple_cb){.count = 0, .ptr = unspecified_ptr};
}

inline void inline_header_compound_literal_unspecified_cast_0(struct simple_cb s) {
  s = (struct simple_cb){.count = 0, .ptr = (int*)0};
}

inline void inline_header_compound_literal_unsafe_indexable_cast_0(struct simple_cb s) {
  s = (struct simple_cb){.count = 0, .ptr = (int* __unsafe_indexable)0};
}

inline void inline_header_compound_literal_unsafe_indexable_ptr(struct simple_cb s, int* __unsafe_indexable unsafe_indexable_ptr) {
  // strict-error@+2{{initializing 'int *__single __counted_by(count)' (aka 'int *__single') with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // incorrect-error@+1{{cannot extract the lower bound of 'int *__unsafe_indexable' because it has no bounds specification}}
  s = (struct simple_cb){.count = 0, .ptr = unsafe_indexable_ptr};
}
