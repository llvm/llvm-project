// A collection of various initializers which shouldn't trip up initialization
// order checking.  If successful, this will just return 0.

// RUN: %clangxx_asan -O0 %s %p/Helpers/initialization-nobug-extra.cpp -o %t
// RUN: %env_asan_opts=check_initialization_order=true %run %t 2>&1
// RUN: %clangxx_asan -O1 %s %p/Helpers/initialization-nobug-extra.cpp -o %t
// RUN: %env_asan_opts=check_initialization_order=true %run %t 2>&1
// RUN: %clangxx_asan -O2 %s %p/Helpers/initialization-nobug-extra.cpp -o %t
// RUN: %env_asan_opts=check_initialization_order=true %run %t 2>&1
// RUN: %clangxx_asan -O3 %s %p/Helpers/initialization-nobug-extra.cpp -o %t
// RUN: %env_asan_opts=check_initialization_order=true %run %t 2>&1

// Simple access:
// Make sure that accessing a global in the same TU is safe

bool condition = true;
int initializeSameTU() {
  return condition ? 0x2a : 052;
}
int sameTU = initializeSameTU();

// Linker initialized:
// Check that access to linker initialized globals originating from a different
// TU's initializer is safe.

int A = (1 << 1) + (1 << 3) + (1 << 5), B;
int getAB() {
  return A * B;
}

// Function local statics:
// Check that access to function local statics originating from a different
// TU's initializer is safe.

int countCalls() {
  static int calls;
  return ++calls;
}

// Trivial constructor, non-trivial destructor.
struct StructWithDtor {
  ~StructWithDtor() { }
  int value;
};
StructWithDtor struct_with_dtor;
int getStructWithDtorValue() { return struct_with_dtor.value; }

int main() { return 0; }
