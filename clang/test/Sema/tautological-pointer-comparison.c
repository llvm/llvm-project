// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fwrapv -verify=fwrapv %s

// fwrapv-no-diagnostics

int add_ptr_idx_ult_ptr(const char *ptr, unsigned index) {
  return ptr + index < ptr; // expected-warning {{pointer comparison always evaluates to false}}
}

int add_idx_ptr_ult_ptr(const char *ptr, unsigned index) {
  return index + ptr < ptr; // expected-warning {{pointer comparison always evaluates to false}}
}

int ptr_ugt_add_ptr_idx(const char *ptr, unsigned index) {
  return ptr > ptr + index; // expected-warning {{pointer comparison always evaluates to false}}
}

int ptr_ugt_add_idx_ptr(const char *ptr, unsigned index) {
  return ptr > index + ptr; // expected-warning {{pointer comparison always evaluates to false}}
}

int add_ptr_idx_uge_ptr(const char *ptr, unsigned index) {
  return ptr + index >= ptr; // expected-warning {{pointer comparison always evaluates to true}}
}

int add_idx_ptr_uge_ptr(const char *ptr, unsigned index) {
  return index + ptr >= ptr; // expected-warning {{pointer comparison always evaluates to true}}
}

int ptr_ule_add_ptr_idx(const char *ptr, unsigned index) {
  return ptr <= ptr + index; // expected-warning {{pointer comparison always evaluates to true}}
}

int ptr_ule_add_idx_ptr(const char *ptr, unsigned index) {
  return ptr <= index + ptr; // expected-warning {{pointer comparison always evaluates to true}}
}

int add_ptr_idx_ult_ptr_array(unsigned index) {
  char ptr[10];
  return ptr + index < ptr; // expected-warning {{pointer comparison always evaluates to false}}
}

// Negative tests with wrong predicate.

int add_ptr_idx_ule_ptr(const char *ptr, unsigned index) {
  return ptr + index <= ptr;
}

int add_ptr_idx_ugt_ptr(const char *ptr, unsigned index) {
  return ptr + index > ptr;
}

int ptr_uge_add_idx_ptr(const char *ptr, unsigned index) {
  return ptr >= index + ptr;
}

int ptr_ult_add_idx_ptr(const char *ptr, unsigned index) {
  return ptr < index + ptr;
}

// Negative test with signed index.

int add_ptr_idx_ult_ptr_signed(const char *ptr, int index) {
  return ptr + index < ptr;
}

// Negative test with unrelated pointers. 

int add_ptr_idx_ult_ptr2(const char *ptr, const char *ptr2, unsigned index) {
  return ptr + index < ptr2;
}

// Negative test with non-pointer operands.

int add_ptr_idx_ult_ptr_not_pointer(unsigned ptr, unsigned index) {
  return ptr + index < ptr;
}
