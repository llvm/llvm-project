// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-implicit-function-declaration -w -o - -emit-llvm %s | FileCheck %s -check-prefix=NO__ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-implicit-function-declaration -w -o - -emit-llvm -fmath-errno %s | FileCheck %s -check-prefix=HAS_ERRNO

// Test attributes and builtin codegen of complex library calls.

void foo(float f) {
  cabs(f);       cabsf(f);      cabsl(f);

// NO__ERRNO: declare double @cabs(double noundef, double noundef) [[READNONE:#[0-9]+]]
// NO__ERRNO: declare float @cabsf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cabsl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare double @cabs(double noundef, double noundef) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @cabsf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cabsl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cacos(f);      cacosf(f);     cacosl(f);

// NO__ERRNO: declare { double, double } @cacos(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacosf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacos(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacosf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cacosh(f);     cacoshf(f);    cacoshl(f);

// NO__ERRNO: declare { double, double } @cacosh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacoshf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacosh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacoshf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  carg(f);       cargf(f);      cargl(f);

// NO__ERRNO: declare double @carg(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare float @cargf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cargl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare double @carg(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @cargf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cargl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  casin(f);      casinf(f);     casinl(f);

// NO__ERRNO: declare { double, double } @casin(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casin(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  casinh(f);     casinhf(f);    casinhl(f);

// NO__ERRNO: declare { double, double } @casinh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinhf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casinh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinhf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  catan(f);      catanf(f);     catanl(f);

// NO__ERRNO: declare { double, double } @catan(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catan(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  catanh(f);     catanhf(f);    catanhl(f);

// NO__ERRNO: declare { double, double } @catanh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanhf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catanh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanhf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ccos(f);       ccosf(f);      ccosl(f);

// NO__ERRNO: declare { double, double } @ccos(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccosf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccos(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccosf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ccosh(f);      ccoshf(f);     ccoshl(f);

// NO__ERRNO: declare { double, double } @ccosh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccoshf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccosh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccoshf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cexp(f);       cexpf(f);      cexpl(f);

// NO__ERRNO: declare { double, double } @cexp(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cexpf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cexpl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cexp(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cexpf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cexpl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cimag(f);      cimagf(f);     cimagl(f);

// NO__ERRNO-NOT: .cimag
// NO__ERRNO-NOT: @cimag
// HAS_ERRNO-NOT: .cimag
// HAS_ERRNO-NOT: @cimag

  conj(f);       conjf(f);      conjl(f);

// NO__ERRNO-NOT: .conj
// NO__ERRNO-NOT: @conj
// HAS_ERRNO-NOT: .conj
// HAS_ERRNO-NOT: @conj

  clog(f);       clogf(f);      clogl(f);

// NO__ERRNO: declare { double, double } @clog(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @clogf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @clogl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @clog(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @clogf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @clogl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cproj(f);      cprojf(f);     cprojl(f);

// NO__ERRNO: declare { double, double } @cproj(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cprojf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cprojl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cproj(double noundef, double noundef) [[READNONE:#[0-9]+]]
// HAS_ERRNO: declare <2 x float> @cprojf(<2 x float> noundef) [[READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cprojl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[WILLRETURN_NOT_READNONE:#[0-9]+]]

  cpow(f,f);       cpowf(f,f);      cpowl(f,f);

// NO__ERRNO: declare { double, double } @cpow(double noundef, double noundef, double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cpowf(<2 x float> noundef, <2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cpowl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16, ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cpow(double noundef, double noundef, double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cpowf(<2 x float> noundef, <2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cpowl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16, ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  creal(f);      crealf(f);     creall(f);

// NO__ERRNO-NOT: .creal
// NO__ERRNO-NOT: @creal
// HAS_ERRNO-NOT: .creal
// HAS_ERRNO-NOT: @creal

  csin(f);       csinf(f);      csinl(f);

// NO__ERRNO: declare { double, double } @csin(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csin(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  csinh(f);      csinhf(f);     csinhl(f);

// NO__ERRNO: declare { double, double } @csinh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinhf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csinh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinhf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  csqrt(f);      csqrtf(f);     csqrtl(f);

// NO__ERRNO: declare { double, double } @csqrt(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csqrtf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csqrt(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csqrtf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ctan(f);       ctanf(f);      ctanl(f);

// NO__ERRNO: declare { double, double } @ctan(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ctanf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ctanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ctan(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ctanf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ctanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ctanh(f);      ctanhf(f);     ctanhl(f);

// NO__ERRNO: declare { double, double } @ctanh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ctanhf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ctanhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ctanh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ctanhf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ctanhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
};

// NO__ERRNO: attributes [[READNONE]] = { {{.*}}memory(none){{.*}} }
// NO__ERRNO: attributes [[NOT_READNONE]] = { nounwind {{.*}} }

// HAS_ERRNO: attributes [[NOT_READNONE]] = { nounwind {{.*}} }
// HAS_ERRNO: attributes [[READNONE]] = { {{.*}}memory(none){{.*}} }
// HAS_ERRNO: attributes [[WILLRETURN_NOT_READNONE]] = { nounwind willreturn {{.*}} }
