// RUN: %clang_cc1 -triple i686-windows-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=GNU32
// RUN: %clang_cc1 -triple i686-windows-gnu -emit-llvm -o - %s -mms-bitfields \
// RUN:    | FileCheck %s --check-prefix=GNU32
// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=GNU64
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=MSC64

struct {
  char c;
  long double ldb;
} agggregate_LD = {};
// GNU32: %struct.anon = type { i8, x86_fp80 }
// GNU32: @agggregate_LD = dso_local global %struct.anon zeroinitializer, align 4
// GNU64: %struct.anon = type { i8, x86_fp80 }
// GNU64: @agggregate_LD = dso_local global %struct.anon zeroinitializer, align 16
// MSC64: %struct.anon = type { i8, double }
// MSC64: @agggregate_LD = dso_local global %struct.anon zeroinitializer, align 8

long double dataLD = 1.0L;
// GNU32: @dataLD = dso_local global x86_fp80 0xK3FFF8000000000000000, align 4
// GNU64: @dataLD = dso_local global x86_fp80 0xK3FFF8000000000000000, align 16
// MSC64: @dataLD = dso_local global double 1.000000e+00, align 8

long double _Complex dataLDC = {1.0L, 1.0L};
// GNU32: @dataLDC = dso_local global { x86_fp80, x86_fp80 } { x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK3FFF8000000000000000 }, align 4
// GNU64: @dataLDC = dso_local global { x86_fp80, x86_fp80 } { x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK3FFF8000000000000000 }, align 16
// MSC64: @dataLDC = dso_local global { double, double } { double 1.000000e+00, double 1.000000e+00 }, align 8

long double TestLD(long double x) {
  return x * x;
}
// GNU32: define dso_local x86_fp80 @TestLD(x86_fp80 noundef %x)
// GNU64: define dso_local void @TestLD(ptr dead_on_unwind noalias writable sret(x86_fp80) align 16 %agg.result, ptr noundef %0)
// MSC64: define dso_local double @TestLD(double noundef %x)

long double _Complex TestLDC(long double _Complex x) {
  return x * x;
}
// GNU32: define dso_local void @TestLDC(ptr dead_on_unwind noalias writable sret({ x86_fp80, x86_fp80 }) align 4 %agg.result, ptr noundef byval({ x86_fp80, x86_fp80 }) align 4 %x)
// GNU64: define dso_local void @TestLDC(ptr dead_on_unwind noalias writable sret({ x86_fp80, x86_fp80 }) align 16 %agg.result, ptr noundef %x)
// MSC64: define dso_local void @TestLDC(ptr dead_on_unwind noalias writable sret({ double, double }) align 8 %agg.result, ptr noundef %x)

// GNU32: declare dso_local void @__mulxc3
// GNU64: declare dso_local void @__mulxc3
// MSC64: declare dso_local void @__muldc3

void VarArgLD(int a, ...) {
  // GNU32-LABEL: define{{.*}} void @VarArgLD
  // GNU64-LABEL: define{{.*}} void @VarArgLD
  // MSC64-LABEL: define{{.*}} void @VarArgLD
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  long double LD = __builtin_va_arg(ap, long double);
  // GNU32: load x86_fp80, ptr %argp.cur
  // GNU64: [[P:%.*]] = load ptr, ptr %argp.cur
  // GNU64: load x86_fp80, ptr [[P]]
  // MSC64: load double, ptr %argp.cur
  __builtin_va_end(ap);
}
