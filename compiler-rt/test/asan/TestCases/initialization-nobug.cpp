// A collection of various initializers which shouldn't trip up initialization
// order checking.  If successful, this will just return 0.

// RUN: %clangxx_asan -O0 %s %p/Helpers/initialization-nobug-extra.cpp -o %t && %env_asan_opts=check_initialization_order=true:report_globals=3 %run %t 2>&1 | FileCheck %s --implicit-check-not "DynInitPoison"
// RUN: %clangxx_asan -O1 %s %p/Helpers/initialization-nobug-extra.cpp -o %t && %env_asan_opts=check_initialization_order=true:report_globals=3 %run %t 2>&1 | FileCheck %s --implicit-check-not "DynInitPoison"
// RUN: %clangxx_asan -O2 %s %p/Helpers/initialization-nobug-extra.cpp -o %t && %env_asan_opts=check_initialization_order=true:report_globals=3 %run %t 2>&1 | FileCheck %s --implicit-check-not "DynInitPoison"
// RUN: %clangxx_asan -O3 %s %p/Helpers/initialization-nobug-extra.cpp -o %t && %env_asan_opts=check_initialization_order=true:report_globals=3 %run %t 2>&1 | FileCheck %s --implicit-check-not "DynInitPoison"

// Simple access:
// Make sure that accessing a global in the same TU is safe

bool condition = true;
__attribute__((noinline, weak)) int initializeSameTU() {
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

// CHECK: DynInitPoison
// CHECK: DynInitPoison

// In general case entire set of DynInitPoison must be followed by at lest one
// DynInitUnpoison. In some cases we can limit the number of DynInitUnpoison,
// see initialization-nobug-lld.cpp.

// CHECK: DynInitUnpoison
