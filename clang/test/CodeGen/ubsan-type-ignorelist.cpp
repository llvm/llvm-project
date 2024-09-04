// Verify ubsan doesn't emit checks for ignorelisted types
// RUN: echo "[{unsigned-integer-overflow,signed-integer-overflow}]" > %t-int.ignorelist
// RUN: echo "type:int" >> %t-int.ignorelist
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-ignorelist=%t-int.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=INT

// RUN: echo "type:int" > %t-nosection.ignorelist
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-ignorelist=%t-nosection.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=INT

// RUN: echo "type:int=allow" > %t-allow-same-as-no-category.ignorelist
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-ignorelist=%t-allow-same-as-no-category.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=INT

// RUN: echo "[{unsigned-integer-overflow,signed-integer-overflow}]" > %t-myty.ignorelist
// RUN: echo "type:*" >> %t-myty.ignorelist
// RUN: echo "type:myty=skip" >> %t-myty.ignorelist
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-ignorelist=%t-myty.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=MYTY

// RUN: echo "[{implicit-signed-integer-truncation,implicit-unsigned-integer-truncation}]" > %t-trunc.ignorelist
// RUN: echo "type:char" >> %t-trunc.ignorelist
// RUN: echo "type:unsigned char" >> %t-trunc.ignorelist
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=implicit-signed-integer-truncation,implicit-unsigned-integer-truncation -fsanitize-ignorelist=%t-trunc.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=TRUNC

// Verify ubsan vptr does not check down-casts on ignorelisted types.
// RUN: echo "type:_ZTI3Foo" > %t-type.ignorelist
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=vptr -fsanitize-recover=vptr -emit-llvm %s -o - | FileCheck %s --check-prefix=VPTR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=vptr -fsanitize-recover=vptr -fsanitize-ignorelist=%t-type.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=VPTR-TYPE

class Bar {
public:
  virtual ~Bar() {}
};
class Foo : public Bar {};

Bar bar;

// VPTR: @_Z7checkmev
// VPTR-TYPE: @_Z7checkmev
void checkme() {
// VPTR: call void @__ubsan_handle_dynamic_type_cache_miss({{.*}} (ptr @bar to
// VPTR-TYPE-NOT: @__ubsan_handle_dynamic_type_cache_miss
  Foo* foo = static_cast<Foo*>(&bar); // down-casting
// VPTR: ret void
// VPTR-TYPE: ret void
  return;
}

// INT-LABEL: ignore_int
void ignore_int(int A, int B, unsigned C, unsigned D, long E) {
// INT: llvm.uadd.with.overflow.i32
  (void)(C+D);
// INT-NOT: llvm.sadd.with.overflow.i32
  (void)(A+B);
// INT: llvm.sadd.with.overflow.i64
  (void)(++E);
}


typedef unsigned long myty;
typedef myty derivative;
// INT-LABEL: ignore_all_except_myty
// MYTY-LABEL: ignore_all_except_myty
void ignore_all_except_myty(myty A, myty B, int C, unsigned D, derivative E) {
// MYTY-NOT: llvm.sadd.with.overflow.i32
  (void)(++C);

// MYTY-NOT: llvm.uadd.with.overflow.i32
  (void)(D+D);

// MYTY-NOT: llvm.umul.with.overflow.i64
  (void)(E*2);

// MYTY: llvm.uadd.with.overflow.i64
  (void)(A+B);
}

// INT-LABEL: truncation
// MYTY-LABEL: truncation
// TRUNC-LABEL: truncation
void truncation(char A, int B, unsigned char C, short D) {
// TRUNC-NOT: %handler.implicit_conversion
  A = B;
// TRUNC-NOT: %handler.implicit_conversion
  A = C;
// TRUNC-NOT: %handler.implicit_conversion
  C = B;

// TRUNC: %handler.implicit_conversion
  D = B;

  (void)A;
  (void)D;
}
