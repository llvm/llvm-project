// RUN: %clang_cc1 -E -dM %s | FileCheck --strict-whitespace --check-prefix=NOMS %s
// RUN: %clang_cc1 -fms-compatibility -E -dM %s | FileCheck --strict-whitespace --check-prefix=MS %s

// If the assert macro is defined in MS compatibility mode in C, we
// automatically inject a macro definition for static_assert. Test that the
// macro is properly added to the preprocessed output. This allows us to
// diagonse use of the static_assert keyword when <assert.h> has not been
// included while still being able to compile preprocessed code.
#define assert

MS: #define static_assert _Static_assert
NOMS-NOT: #define static_assert _Static_assert
