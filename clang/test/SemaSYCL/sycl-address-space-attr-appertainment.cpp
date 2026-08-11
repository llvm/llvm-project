// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -std=c++20 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++20 -verify %s

// The SYCL address space attributes are type attributes that may only appear on
// the object type of an object pointer or object reference type. This test
// exercises that appropriate diagnostics are issued for other uses.

//===----------------------------------------------------------------------===//
// sycl_global
//===----------------------------------------------------------------------===//

// expected-error@+1{{'clang::sycl_global' attribute cannot be applied to a declaration}}
[[clang::sycl_global]];

// expected-error@+1{{'clang::sycl_global' attribute cannot be applied to a declaration}}
namespace [[clang::sycl_global]] global_ns {}

// expected-error@+1{{'clang::sycl_global' attribute cannot be applied to a declaration}}
struct [[clang::sycl_global]] global_struct {};

// expected-error@+1{{'clang::sycl_global' attribute cannot be applied to a declaration}}
enum [[clang::sycl_global]] global_enum {};

enum {
  // expected-error@+1{{'clang::sycl_global' attribute cannot be applied to a declaration}}
  global_enumerator [[clang::sycl_global]]
};

template <typename>
// expected-error@+1{{'clang::sycl_global' attribute cannot be applied to a declaration}}
concept global_concept [[clang::sycl_global]] = true;

// expected-error@+1{{function type may not be qualified with an address space}}
[[clang::sycl_global]] int global_ret();

// expected-error@+1{{function type may not be qualified with an address space}}
int global_fn_param(void (fp [[clang::sycl_global]])());

// expected-error@+1{{function type may not be qualified with an address space}}
int global_trailing() [[clang::sycl_global]];

struct global_members {
  // expected-error@+1{{function type may not be qualified with an address space}}
  [[clang::sycl_global]] int mf();
  // expected-error@+2{{field may not be qualified with an address space}}
  // expected-error@+1{{'clang::sycl_global' attribute cannot be applied to a declaration}}
  [[clang::sycl_global]] int dm;
};

// expected-error@+1{{'[[clang::sycl_global]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_global]] global_object;

// expected-error@+1{{'[[clang::sycl_global]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int * [[clang::sycl_global]] global_pointer_object;

// expected-error@+1{{'[[clang::sycl_global]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_global]] global_array[4];

using global_int = int [[clang::sycl_global]];
// expected-error@+1{{'[[clang::sycl_global]]' attribute may only be applied to the object type of an object pointer or object reference type}}
global_int global_object_via_typedef;

void global_storage_durations() {
  // expected-error@+1{{'[[clang::sycl_global]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  int [[clang::sycl_global]] automatic_object;
  // expected-error@+1{{'[[clang::sycl_global]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  static int [[clang::sycl_global]] static_object;
}

struct global_S;
int [[clang::sycl_global]] global_S::*global_member_pointer;
using global_int_ptr = global_int *;
global_int_ptr global_pointer_via_typedef;
int [[clang::sycl_global]] *global_pointer;
void global_reference(int [[clang::sycl_global]] &r);
int [[clang::sycl_global]] *global_returns_pointer();

//===----------------------------------------------------------------------===//
// sycl_local
//===----------------------------------------------------------------------===//

// expected-error@+1{{'clang::sycl_local' attribute cannot be applied to a declaration}}
[[clang::sycl_local]];

// expected-error@+1{{'clang::sycl_local' attribute cannot be applied to a declaration}}
namespace [[clang::sycl_local]] local_ns {}

// expected-error@+1{{'clang::sycl_local' attribute cannot be applied to a declaration}}
struct [[clang::sycl_local]] local_struct {};

// expected-error@+1{{'clang::sycl_local' attribute cannot be applied to a declaration}}
enum [[clang::sycl_local]] local_enum {};

enum {
  // expected-error@+1{{'clang::sycl_local' attribute cannot be applied to a declaration}}
  local_enumerator [[clang::sycl_local]]
};

template <typename>
// expected-error@+1{{'clang::sycl_local' attribute cannot be applied to a declaration}}
concept local_concept [[clang::sycl_local]] = true;

// expected-error@+1{{function type may not be qualified with an address space}}
[[clang::sycl_local]] int local_ret();

// expected-error@+1{{function type may not be qualified with an address space}}
int local_fn_param(void (fp [[clang::sycl_local]])());

// expected-error@+1{{function type may not be qualified with an address space}}
int local_trailing() [[clang::sycl_local]];

struct local_members {
  // expected-error@+1{{function type may not be qualified with an address space}}
  [[clang::sycl_local]] int mf();
  // expected-error@+2{{field may not be qualified with an address space}}
  // expected-error@+1{{'clang::sycl_local' attribute cannot be applied to a declaration}}
  [[clang::sycl_local]] int dm;
};

// expected-error@+1{{'[[clang::sycl_local]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_local]] local_object;

// expected-error@+1{{'[[clang::sycl_local]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int * [[clang::sycl_local]] local_pointer_object;

// expected-error@+1{{'[[clang::sycl_local]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_local]] local_array[4];

using local_int = int [[clang::sycl_local]];
// expected-error@+1{{'[[clang::sycl_local]]' attribute may only be applied to the object type of an object pointer or object reference type}}
local_int local_object_via_typedef;

void local_storage_durations() {
  // expected-error@+1{{'[[clang::sycl_local]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  int [[clang::sycl_local]] automatic_object;
  // expected-error@+1{{'[[clang::sycl_local]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  static int [[clang::sycl_local]] static_object;
}

struct local_S;
int [[clang::sycl_local]] local_S::*local_member_pointer;
using local_int_ptr = local_int *;
local_int_ptr local_pointer_via_typedef;
int [[clang::sycl_local]] *local_pointer;
void local_reference(int [[clang::sycl_local]] &r);
int [[clang::sycl_local]] *local_returns_pointer();

//===----------------------------------------------------------------------===//
// sycl_private
//===----------------------------------------------------------------------===//

// expected-error@+1{{'clang::sycl_private' attribute cannot be applied to a declaration}}
[[clang::sycl_private]];

// expected-error@+1{{'clang::sycl_private' attribute cannot be applied to a declaration}}
namespace [[clang::sycl_private]] private_ns {}

// expected-error@+1{{'clang::sycl_private' attribute cannot be applied to a declaration}}
struct [[clang::sycl_private]] private_struct {};

// expected-error@+1{{'clang::sycl_private' attribute cannot be applied to a declaration}}
enum [[clang::sycl_private]] private_enum {};

enum {
  // expected-error@+1{{'clang::sycl_private' attribute cannot be applied to a declaration}}
  private_enumerator [[clang::sycl_private]]
};

template <typename>
// expected-error@+1{{'clang::sycl_private' attribute cannot be applied to a declaration}}
concept private_concept [[clang::sycl_private]] = true;

// expected-error@+1{{function type may not be qualified with an address space}}
[[clang::sycl_private]] int private_ret();

// expected-error@+1{{function type may not be qualified with an address space}}
int private_fn_param(void (fp [[clang::sycl_private]])());

// expected-error@+1{{function type may not be qualified with an address space}}
int private_trailing() [[clang::sycl_private]];

struct private_members {
  // expected-error@+1{{function type may not be qualified with an address space}}
  [[clang::sycl_private]] int mf();
  // expected-error@+2{{field may not be qualified with an address space}}
  // expected-error@+1{{'clang::sycl_private' attribute cannot be applied to a declaration}}
  [[clang::sycl_private]] int dm;
};

// expected-error@+1{{'[[clang::sycl_private]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_private]] private_object;

// expected-error@+1{{'[[clang::sycl_private]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int * [[clang::sycl_private]] private_pointer_object;

// expected-error@+1{{'[[clang::sycl_private]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_private]] private_array[4];

using private_int = int [[clang::sycl_private]];
// expected-error@+1{{'[[clang::sycl_private]]' attribute may only be applied to the object type of an object pointer or object reference type}}
private_int private_object_via_typedef;

void private_storage_durations() {
  // expected-error@+1{{'[[clang::sycl_private]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  int [[clang::sycl_private]] automatic_object;
  // expected-error@+1{{'[[clang::sycl_private]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  static int [[clang::sycl_private]] static_object;
}

struct private_S;
int [[clang::sycl_private]] private_S::*private_member_pointer;
using private_int_ptr = private_int *;
private_int_ptr private_pointer_via_typedef;
int [[clang::sycl_private]] *private_pointer;
void private_reference(int [[clang::sycl_private]] &r);
int [[clang::sycl_private]] *private_returns_pointer();

//===----------------------------------------------------------------------===//
// sycl_generic
//===----------------------------------------------------------------------===//

// expected-error@+1{{'clang::sycl_generic' attribute cannot be applied to a declaration}}
[[clang::sycl_generic]];

// expected-error@+1{{'clang::sycl_generic' attribute cannot be applied to a declaration}}
namespace [[clang::sycl_generic]] generic_ns {}

// expected-error@+1{{'clang::sycl_generic' attribute cannot be applied to a declaration}}
struct [[clang::sycl_generic]] generic_struct {};

// expected-error@+1{{'clang::sycl_generic' attribute cannot be applied to a declaration}}
enum [[clang::sycl_generic]] generic_enum {};

enum {
  // expected-error@+1{{'clang::sycl_generic' attribute cannot be applied to a declaration}}
  generic_enumerator [[clang::sycl_generic]]
};

template <typename>
// expected-error@+1{{'clang::sycl_generic' attribute cannot be applied to a declaration}}
concept generic_concept [[clang::sycl_generic]] = true;

// expected-error@+1{{function type may not be qualified with an address space}}
[[clang::sycl_generic]] int generic_ret();

// expected-error@+1{{function type may not be qualified with an address space}}
int generic_fn_param(void (fp [[clang::sycl_generic]])());

// expected-error@+1{{function type may not be qualified with an address space}}
int generic_trailing() [[clang::sycl_generic]];

struct generic_members {
  // expected-error@+1{{function type may not be qualified with an address space}}
  [[clang::sycl_generic]] int mf();
  // expected-error@+2{{field may not be qualified with an address space}}
  // expected-error@+1{{'clang::sycl_generic' attribute cannot be applied to a declaration}}
  [[clang::sycl_generic]] int dm;
};

// expected-error@+1{{'[[clang::sycl_generic]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_generic]] generic_object;

// expected-error@+1{{'[[clang::sycl_generic]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int * [[clang::sycl_generic]] generic_pointer_object;

// expected-error@+1{{'[[clang::sycl_generic]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_generic]] generic_array[4];

using generic_int = int [[clang::sycl_generic]];
// expected-error@+1{{'[[clang::sycl_generic]]' attribute may only be applied to the object type of an object pointer or object reference type}}
generic_int generic_object_via_typedef;

void generic_storage_durations() {
  // expected-error@+1{{'[[clang::sycl_generic]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  int [[clang::sycl_generic]] automatic_object;
  // expected-error@+1{{'[[clang::sycl_generic]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  static int [[clang::sycl_generic]] static_object;
}

struct generic_S;
int [[clang::sycl_generic]] generic_S::*generic_member_pointer;
using generic_int_ptr = generic_int *;
generic_int_ptr generic_pointer_via_typedef;
int [[clang::sycl_generic]] *generic_pointer;
void generic_reference(int [[clang::sycl_generic]] &r);
int [[clang::sycl_generic]] *generic_returns_pointer();

//===----------------------------------------------------------------------===//
// sycl_constant
//===----------------------------------------------------------------------===//

// expected-error@+1{{'clang::sycl_constant' attribute cannot be applied to a declaration}}
[[clang::sycl_constant]];

// expected-error@+1{{'clang::sycl_constant' attribute cannot be applied to a declaration}}
namespace [[clang::sycl_constant]] constant_ns {}

// expected-error@+1{{'clang::sycl_constant' attribute cannot be applied to a declaration}}
struct [[clang::sycl_constant]] constant_struct {};

// expected-error@+1{{'clang::sycl_constant' attribute cannot be applied to a declaration}}
enum [[clang::sycl_constant]] constant_enum {};

enum {
  // expected-error@+1{{'clang::sycl_constant' attribute cannot be applied to a declaration}}
  constant_enumerator [[clang::sycl_constant]]
};

template <typename>
// expected-error@+1{{'clang::sycl_constant' attribute cannot be applied to a declaration}}
concept constant_concept [[clang::sycl_constant]] = true;

// expected-error@+1{{function type may not be qualified with an address space}}
[[clang::sycl_constant]] int constant_ret();

// expected-error@+1{{function type may not be qualified with an address space}}
int constant_fn_param(void (fp [[clang::sycl_constant]])());

// expected-error@+1{{function type may not be qualified with an address space}}
int constant_trailing() [[clang::sycl_constant]];

struct constant_members {
  // expected-error@+1{{function type may not be qualified with an address space}}
  [[clang::sycl_constant]] int mf();
  // expected-error@+2{{field may not be qualified with an address space}}
  // expected-error@+1{{'clang::sycl_constant' attribute cannot be applied to a declaration}}
  [[clang::sycl_constant]] int dm;
};

// expected-error@+1{{'[[clang::sycl_constant]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_constant]] constant_object;

// expected-error@+1{{'[[clang::sycl_constant]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int * [[clang::sycl_constant]] constant_pointer_object;

// expected-error@+1{{'[[clang::sycl_constant]]' attribute may only be applied to the object type of an object pointer or object reference type}}
int [[clang::sycl_constant]] constant_array[4];

using constant_int = int [[clang::sycl_constant]];
// expected-error@+1{{'[[clang::sycl_constant]]' attribute may only be applied to the object type of an object pointer or object reference type}}
constant_int constant_object_via_typedef;

void constant_storage_durations() {
  // expected-error@+1{{'[[clang::sycl_constant]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  int [[clang::sycl_constant]] automatic_object;
  // expected-error@+1{{'[[clang::sycl_constant]]' attribute may only be applied to the object type of an object pointer or object reference type}}
  static int [[clang::sycl_constant]] static_object;
}

struct constant_S;
int [[clang::sycl_constant]] constant_S::*constant_member_pointer;
using constant_int_ptr = constant_int *;
constant_int_ptr constant_pointer_via_typedef;
int [[clang::sycl_constant]] *constant_pointer;
void constant_reference(int [[clang::sycl_constant]] &r);
int [[clang::sycl_constant]] *constant_returns_pointer();
