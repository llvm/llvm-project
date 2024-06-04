// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm %s | FileCheck %s -check-prefix=NO__ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm -fmath-errno %s | FileCheck %s -check-prefix=HAS_ERRNO

// Test attributes and codegen of complex builtins.

void foo(float f) {
  __builtin_cabs(f);       __builtin_cabsf(f);      __builtin_cabsl(f);

// NO__ERRNO: declare double @cabs(double noundef, double noundef) [[READNONE:#[0-9]+]]
// NO__ERRNO: declare float @cabsf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cabsl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare double @cabs(double noundef, double noundef) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @cabsf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cabsl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_cacos(f);      __builtin_cacosf(f);     __builtin_cacosl(f);

// NO__ERRNO: declare { double, double } @cacos(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacosf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacos(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacosf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_cacosh(f);     __builtin_cacoshf(f);    __builtin_cacoshl(f);

// NO__ERRNO: declare { double, double } @cacosh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacoshf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacosh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacoshf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_carg(f);       __builtin_cargf(f);      __builtin_cargl(f);

// NO__ERRNO: declare double @carg(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare float @cargf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cargl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare double @carg(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @cargf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cargl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_casin(f);      __builtin_casinf(f);     __builtin_casinl(f);

// NO__ERRNO: declare { double, double } @casin(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casin(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_casinh(f);     __builtin_casinhf(f);    __builtin_casinhl(f); 

// NO__ERRNO: declare { double, double } @casinh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinhf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casinh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinhf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_catan(f);      __builtin_catanf(f);     __builtin_catanl(f); 

// NO__ERRNO: declare { double, double } @catan(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catan(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_catanh(f);     __builtin_catanhf(f);    __builtin_catanhl(f);

// NO__ERRNO: declare { double, double } @catanh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanhf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catanh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanhf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_ccos(f);       __builtin_ccosf(f);      __builtin_ccosl(f);

// NO__ERRNO: declare { double, double } @ccos(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccosf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccos(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccosf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccosl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_ccosh(f);      __builtin_ccoshf(f);     __builtin_ccoshl(f);

// NO__ERRNO: declare { double, double } @ccosh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccoshf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccosh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccoshf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_cexp(f);       __builtin_cexpf(f);      __builtin_cexpl(f);

// NO__ERRNO: declare { double, double } @cexp(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cexpf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cexpl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cexp(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cexpf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cexpl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_cimag(f);      __builtin_cimagf(f);     __builtin_cimagl(f);

// NO__ERRNO-NOT: .cimag
// NO__ERRNO-NOT: @cimag
// HAS_ERRNO-NOT: .cimag
// HAS_ERRNO-NOT: @cimag

  __builtin_conj(f);       __builtin_conjf(f);      __builtin_conjl(f);

// NO__ERRNO-NOT: .conj
// NO__ERRNO-NOT: @conj
// HAS_ERRNO-NOT: .conj
// HAS_ERRNO-NOT: @conj

  __builtin_clog(f);       __builtin_clogf(f);      __builtin_clogl(f);

// NO__ERRNO: declare { double, double } @clog(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @clogf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @clogl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @clog(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @clogf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @clogl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_cproj(f);      __builtin_cprojf(f);     __builtin_cprojl(f); 

// NO__ERRNO: declare { double, double } @cproj(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cprojf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cprojl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cproj(double noundef, double noundef) [[READNONE:#[0-9]+]]
// HAS_ERRNO: declare <2 x float> @cprojf(<2 x float> noundef) [[READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cprojl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[WILLRETURN_NOT_READNONE:#[0-9]+]]

  __builtin_cpow(f,f);       __builtin_cpowf(f,f);      __builtin_cpowl(f,f);

// NO__ERRNO: declare { double, double } @cpow(double noundef, double noundef, double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cpowf(<2 x float> noundef, <2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cpowl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16, ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cpow(double noundef, double noundef, double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cpowf(<2 x float> noundef, <2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cpowl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16, ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_creal(f);      __builtin_crealf(f);     __builtin_creall(f);

// NO__ERRNO-NOT: .creal
// NO__ERRNO-NOT: @creal
// HAS_ERRNO-NOT: .creal
// HAS_ERRNO-NOT: @creal

  __builtin_csin(f);       __builtin_csinf(f);      __builtin_csinl(f);

// NO__ERRNO: declare { double, double } @csin(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csin(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_csinh(f);      __builtin_csinhf(f);     __builtin_csinhl(f);

// NO__ERRNO: declare { double, double } @csinh(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinhf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csinh(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinhf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinhl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_csqrt(f);      __builtin_csqrtf(f);     __builtin_csqrtl(f);  

// NO__ERRNO: declare { double, double } @csqrt(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csqrtf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csqrt(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csqrtf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_ctan(f);       __builtin_ctanf(f);      __builtin_ctanl(f);

// NO__ERRNO: declare { double, double } @ctan(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ctanf(<2 x float> noundef) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ctanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ctan(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ctanf(<2 x float> noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ctanl(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  __builtin_ctanh(f);      __builtin_ctanhf(f);     __builtin_ctanhl(f); 

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
