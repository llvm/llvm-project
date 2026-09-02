// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// PointerFlow:
//  RUN: %clang_cc1 -fsyntax-only %t/local.cpp \
//  RUN:   --ssaf-extract-summaries=PointerFlow \
//  RUN:   --ssaf-compilation-unit-id=test-cu \
//  RUN:   --ssaf-tu-summary-file=%t/pf.default.json
//  RUN: %clang_cc1 -fsyntax-only %t/local.cpp \
//  RUN:   --ssaf-extract-summaries=PointerFlow \
//  RUN:   --ssaf-include-local-entities \
//  RUN:   --ssaf-compilation-unit-id=test-cu \
//  RUN:   --ssaf-tu-summary-file=%t/pf.with_locals.json

// With the flag, summary_data must gain one additional entity_id entry for
// the local pointer.
// RUN: cat %t/pf.default.json     | grep '"entity_id":' | count 2
// RUN: cat %t/pf.with_locals.json | grep '"entity_id":' | count 3

// UnsafeBufferUsage:
//  RUN: %clang_cc1 -fsyntax-only %t/local.cpp \
//  RUN:   --ssaf-extract-summaries=UnsafeBufferUsage \
//  RUN:   --ssaf-compilation-unit-id=test-cu \
//  RUN:   --ssaf-tu-summary-file=%t/ubu.default.json
//  RUN: %clang_cc1 -fsyntax-only %t/local.cpp \
//  RUN:   --ssaf-extract-summaries=UnsafeBufferUsage \
//  RUN:   --ssaf-include-local-entities \
//  RUN:   --ssaf-compilation-unit-id=test-cu \
//  RUN:   --ssaf-tu-summary-file=%t/ubu.with_locals.json

// The unsafe subscript happens inside the local var's initializer scope so its
// own summary is emitted under the flag.
// RUN: cat %t/ubu.default.json     | grep '"entity_id":' | count 1
// RUN: cat %t/ubu.with_locals.json | grep '"entity_id":' | count 2

//--- local.cpp
// Two externally visible entities: 'g_in', 'g_arr'.
// One local pointer that aliases 'g_in' and is then unsafely subscripted
// in the initializer of 'unsafe_local'.

// Default run: summary_data has exactly 2 entries:
// 'use' and the global pointer's initializer
//
// With-locals run: a third entity_id entry covers the local pointer's
// own contributor scope.

int g_arr[10];
int *g_in = g_arr;

void use(int *param) {
  int *local_ptr = param;
  int unsafe_local = local_ptr[3];
  (void)unsafe_local;
  (void)g_in;
}
