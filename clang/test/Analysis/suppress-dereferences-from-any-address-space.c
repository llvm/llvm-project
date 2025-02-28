// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=core,alpha.core -std=gnu99 -analyzer-config suppress-dereferences-from-any-address-space=false -verify=x86-nosuppress,common %s
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=core,alpha.core -std=gnu99 -verify=x86-suppress,common %s
// RUN: %clang_analyze_cc1 -triple arm-pc-linux-gnu -analyzer-checker=core,alpha.core -std=gnu99 -analyzer-config suppress-dereferences-from-any-address-space=false -verify=other-nosuppress,common %s
// RUN: %clang_analyze_cc1 -triple arm-pc-linux-gnu -analyzer-checker=core,alpha.core -std=gnu99 -verify=other-suppress,common %s

#define AS_ATTRIBUTE(_X) volatile __attribute__((address_space(_X)))

#define _get_base() ((void * AS_ATTRIBUTE(256) *)0)

void* test_address_space_array(unsigned long slot) {
  return _get_base()[slot]; // other-nosuppress-warning{{Dereference}}
}

void test_address_space_condition(int AS_ATTRIBUTE(257) *cpu_data) {
  if (cpu_data == 0) {
    *cpu_data = 3; // other-nosuppress-warning{{Dereference}}
  }
}

struct X { int member; };
int test_address_space_member(void) {
  struct X AS_ATTRIBUTE(258) *data = (struct X AS_ATTRIBUTE(258) *)0UL;
  int ret;
  ret = data->member; // other-nosuppress-warning{{Dereference}}
  return ret;
}

void test_other_address_space_condition(int AS_ATTRIBUTE(259) *cpu_data) {
  if (cpu_data == 0) {
    *cpu_data = 3; // other-nosuppress-warning{{Dereference}} \
                   // x86-nosuppress-warning{{Dereference}}
  }
}

void test_no_address_space_condition(int *cpu_data) {
  if (cpu_data == 0) {
    *cpu_data = 3; // common-warning{{Dereference}}
  }
}

#define _fixed_get_base() ((void * AS_ATTRIBUTE(256) *)2)

void* fixed_test_address_space_array(unsigned long slot) {
  return _fixed_get_base()[slot]; // other-nosuppress-warning{{Dereference}}
}

void fixed_test_address_space_condition(int AS_ATTRIBUTE(257) *cpu_data) {
  if (cpu_data == (int AS_ATTRIBUTE(257) *)2) {
    *cpu_data = 3; // other-nosuppress-warning{{Dereference}}
  }
}

int fixed_test_address_space_member(void) {
  struct X AS_ATTRIBUTE(258) *data = (struct X AS_ATTRIBUTE(258) *)2UL;
  int ret;
  ret = data->member; // other-nosuppress-warning{{Dereference}}
  return ret;
}

void fixed_test_other_address_space_condition(int AS_ATTRIBUTE(259) *cpu_data) {
  if (cpu_data == (int AS_ATTRIBUTE(259) *)2) {
    *cpu_data = 3; // other-nosuppress-warning{{Dereference}} \
                   // x86-nosuppress-warning{{Dereference}}
  }
}

void fixed_test_no_address_space_condition(int *cpu_data) {
  if (cpu_data == (int *)2) {
    *cpu_data = 3; // common-warning{{Dereference}}
  }
}
