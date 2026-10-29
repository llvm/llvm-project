// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-64 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM64EC_LD64
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-80 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM64EC_LD80
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -mlong-double-64 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=X64_LD64
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -mlong-double-80 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=X64_LD80

// The calling convention of a `long double` argument depends on the format
// that `-mlong-double-*` selects, and on the machine.  On ARM64EC, named
// parameters and return values follow the ARM64 convention: a `double`
// occupies a single floating-point register, and the 80-bit image occupies a
// pair of integer registers.  On x86_64 they follow the x64 convention: a
// `double` occupies an XMM register, while the 80-bit image does not fit in
// a register and goes through memory as a reference to a copy and a hidden
// return pointer.  A call across the ARM64EC boundary is translated by an
// entry or an exit thunk, and the two halves below are what such a thunk
// has to translate between.
//
// Variadic arguments follow the x86_64 convention on both machines, where
// anything that is not 1, 2, 4 or 8 bytes is passed by reference: a `double`
// is read back from an 8-byte slot, and the 16-byte image from a reference
// that is 16-byte aligned.
//
// Neither format ever sees the other format's representation.
// ARM64EC_LD64-NOT: alloca i128
// ARM64EC_LD80-NOT: alloca double
// X64_LD64-NOT: alloca x86_fp80
// X64_LD80-NOT: alloca double

typedef __builtin_va_list va_list;

// A named parameter is received in a floating-point register as a `double`,
// or as the 16-byte image in a pair of integer registers; an x86_64 callee
// receives the `double` in an XMM register, and the image through a
// reference.
// ARM64EC_LD64-LABEL: define {{.*}}double @named_receive(double noundef %x)
// ARM64EC_LD80-LABEL: define {{.*}}i128 @named_receive(i128 {{.*}}%x.coerce)
// X64_LD64-LABEL: define {{.*}}double @named_receive(double noundef %x)
// X64_LD80-LABEL: define {{.*}}void @named_receive(ptr dead_on_unwind noalias writable sret(x86_fp80) align 16 %agg.result, ptr {{.*}}align 16{{.*}}%0)
long double named_receive(long double x) { return x; }

// A caller passes a named parameter and receives the return value the same
// way: through a floating-point register, an integer register pair, an XMM
// register, or a reference to a copy of the argument together with a hidden
// return pointer.
// ARM64EC_LD64-LABEL: define {{.*}}double @named_call(double noundef %x)
// ARM64EC_LD64: call double @named_receive(double noundef
// ARM64EC_LD64: ret double
// ARM64EC_LD80-LABEL: define {{.*}}i128 @named_call(i128 {{.*}}%x.coerce)
// ARM64EC_LD80: call i128 @named_receive(i128
// ARM64EC_LD80: ret i128
// X64_LD64-LABEL: define {{.*}}double @named_call(double noundef %x)
// X64_LD64: call double @named_receive(double noundef
// X64_LD64: ret double
// X64_LD80-LABEL: define {{.*}}void @named_call(ptr dead_on_unwind noalias writable sret(x86_fp80) align 16 %agg.result, ptr
// X64_LD80: alloca x86_fp80, align 16
// X64_LD80: store x86_fp80 {{.*}}, ptr {{.*}}, align 16
// X64_LD80: call void @named_receive(ptr dead_on_unwind writable sret(x86_fp80) align 16 {{%.*}}, ptr {{.*}}align 16{{.*}})
long double named_call(long double x) { return named_receive(x); }

int varfmt(const char *fmt, ...);

// A variadic `double` is passed directly, and `va_arg` reads it back from the
// spill slot with 8-byte alignment ...
// ARM64EC_LD64-LABEL: define {{.*}}i32 @vararg_pass(double noundef %x)
// ARM64EC_LD64-NOT: i128
// ARM64EC_LD64: call i32 (ptr, ...) @varfmt(ptr noundef @.str, double noundef
// ARM64EC_LD64-LABEL: define {{.*}}double @vararg_take(ptr noundef %ap)
// ARM64EC_LD64-NOT: i128
// ARM64EC_LD64: load double, ptr %argp.cur, align 8
// X64_LD64-LABEL: define {{.*}}i32 @vararg_pass(double noundef %x)
// X64_LD64-NOT: x86_fp80
// X64_LD64: call i32 (ptr, ...) @varfmt(ptr noundef @.str, double noundef
// X64_LD64-LABEL: define {{.*}}double @vararg_take(ptr noundef %ap)
// X64_LD64-NOT: x86_fp80
// X64_LD64: load double, ptr %argp.cur, align 8
// ... while the 16-byte image is passed by reference to a 16-byte aligned
// temporary, through which `va_arg` loads it.
// ARM64EC_LD80-LABEL: define {{.*}}i32 @vararg_pass(i128 {{.*}}%x.coerce)
// ARM64EC_LD80-NOT: double
// ARM64EC_LD80: alloca x86_fp80, align 16
// ARM64EC_LD80: store x86_fp80 {{.*}}, ptr {{.*}}, align 16
// ARM64EC_LD80: call i32 (ptr, ...) @varfmt(ptr noundef @.str, ptr {{.*}}align 16{{.*}})
// ARM64EC_LD80-LABEL: define {{.*}}i128 @vararg_take(ptr noundef %ap)
// ARM64EC_LD80-NOT: double
// ARM64EC_LD80: load ptr, ptr %argp.cur, align 8
// ARM64EC_LD80: load x86_fp80, ptr {{.*}}, align 16
// ARM64EC_LD80: load i128, ptr %retval, align 16
// X64_LD80-LABEL: define {{.*}}i32 @vararg_pass(ptr {{.*}}align 16{{.*}})
// X64_LD80-NOT: alloca double
// X64_LD80: alloca x86_fp80, align 16
// X64_LD80: store x86_fp80 {{.*}}, ptr {{.*}}, align 16
// X64_LD80: call i32 (ptr, ...) @varfmt(ptr noundef @.str, ptr {{.*}}align 16{{.*}})
// X64_LD80-LABEL: define {{.*}}void @vararg_take(ptr dead_on_unwind noalias writable sret(x86_fp80) align 16 %agg.result, ptr noundef %ap)
// X64_LD80-NOT: alloca double
// X64_LD80: load ptr, ptr %argp.cur, align 8
// X64_LD80: load x86_fp80, ptr {{.*}}, align 16
int vararg_pass(long double x) { return varfmt("%Lg", x); }
long double vararg_take(va_list ap) {
  return __builtin_va_arg(ap, long double);
}

// A return value leaves and enters in a floating-point register, or in a
// pair of integer registers; an x86_64 callee returns the 80-bit image
// through a hidden pointer.
// ARM64EC_LD64-LABEL: define {{.*}}double @ret_out()
// ARM64EC_LD64: ret double 1.000000e+00
// ARM64EC_LD64-LABEL: define {{.*}}double @ret_in()
// ARM64EC_LD64: call double @ret_out()
// ARM64EC_LD80-LABEL: define {{.*}}i128 @ret_out()
// ARM64EC_LD80: store x86_fp80 1.000000e+00, ptr %retval, align 16
// ARM64EC_LD80: load i128, ptr %retval, align 16
// ARM64EC_LD80-LABEL: define {{.*}}i128 @ret_in()
// ARM64EC_LD80: call i128 @ret_out()
// X64_LD64-LABEL: define {{.*}}double @ret_out()
// X64_LD64: ret double 1.000000e+00
// X64_LD64-LABEL: define {{.*}}double @ret_in()
// X64_LD64: call double @ret_out()
// X64_LD80-LABEL: define {{.*}}void @ret_out(ptr dead_on_unwind noalias writable sret(x86_fp80) align 16 %agg.result)
// X64_LD80: store x86_fp80 1.000000e+00, ptr %agg.result, align 16
// X64_LD80-LABEL: define {{.*}}void @ret_in(ptr dead_on_unwind noalias writable sret(x86_fp80) align 16 %agg.result)
// X64_LD80: call void @ret_out(ptr dead_on_unwind writable sret(x86_fp80) align 16
long double ret_out(void) { return 1.0L; }
long double ret_in(void) { return ret_out(); }
