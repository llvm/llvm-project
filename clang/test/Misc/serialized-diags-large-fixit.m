// RUN: %clang -fsyntax-only -fobjc-runtime=macosx-10.8 \
// RUN:   -Wno-objc-root-class --serialize-diagnostics %t.dia %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TEXT
// RUN: c-index-test -read-diagnostics %t.dia 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SERIALIZED

// Form an identifier larger than 64 KiB without making the test file enormous.
#define CAT_IMPL(A, B) A##B
#define CAT(A, B) CAT_IMPL(A, B)
#define A0 a
#define A1 CAT(A0, A0)
#define A2 CAT(A1, A1)
#define A3 CAT(A2, A2)
#define A4 CAT(A3, A3)
#define A5 CAT(A4, A4)
#define A6 CAT(A5, A5)
#define A7 CAT(A6, A6)
#define A8 CAT(A7, A7)
#define A9 CAT(A8, A8)
#define A10 CAT(A9, A9)
#define A11 CAT(A10, A10)
#define A12 CAT(A11, A11)
#define A13 CAT(A12, A12)
#define A14 CAT(A13, A13)
#define A15 CAT(A14, A14)
#define A16 CAT(A15, A15)
#define LARGE_IDENTIFIER CAT(A16, z)

@protocol Protocol
@property int LARGE_IDENTIFIER;
@end

@interface MyClass <Protocol>
@end

@implementation MyClass
@end

// TEXT: warning: auto property synthesis will not synthesize property
// TEXT: note: add a '@synthesize' directive
// SERIALIZED: warning: auto property synthesis will not synthesize property
// SERIALIZED: note: add a '@synthesize' directive
// SERIALIZED: FIXIT:
// SERIALIZED-SAME: z;
// SERIALIZED: Number of diagnostics: 1
