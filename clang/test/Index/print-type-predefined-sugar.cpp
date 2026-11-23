// RUN: c-index-test -test-print-type %s -target x86_64-pc-linux-gnu -std=c++23 | FileCheck %s

typedef struct { void **unused; } S;
void test(S *x, S *y) {
  (void)(x - y);
  (void)sizeof(*x);
  (void)0z;
}

// CHECK: BinaryOperator=- [type=__ptrdiff_t] [typekind=PredefinedSugar] [canonicaltype=long] [canonicaltypekind=Long] [isPOD=1]
// CHECK: UnaryExpr= [type=__size_t] [typekind=PredefinedSugar] [canonicaltype=unsigned long] [canonicaltypekind=ULong] [isPOD=1]
// CHECK: IntegerLiteral= [type=__signed_size_t] [typekind=PredefinedSugar] [canonicaltype=long] [canonicaltypekind=Long] [isPOD=1]
