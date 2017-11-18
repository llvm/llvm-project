// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -S -o - -emit-llvm              %s | FileCheck %s -check-prefix=NO__ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -S -o - -emit-llvm -fmath-errno %s | FileCheck %s -check-prefix=HAS_ERRNO

// Test attributes and codegen of complex builtins.

void foo(float f) {
  __builtin_cabs(f);       __builtin_cabsf(f);      __builtin_cabsl(f);

// NO__ERRNO: declare double @cabs(double, double) [[READNONE:#[0-9]+]]
// NO__ERRNO: declare float @cabsf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cabsl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare double @cabs(double, double) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @cabsf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cabsl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_cacos(f);      __builtin_cacosf(f);     __builtin_cacosl(f);

// NO__ERRNO: declare { double, double } @cacos(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacosf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacosl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacos(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacosf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacosl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_cacosh(f);     __builtin_cacoshf(f);    __builtin_cacoshl(f);

// NO__ERRNO: declare { double, double } @cacosh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacoshf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacosh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacoshf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_carg(f);       __builtin_cargf(f);      __builtin_cargl(f);

// NO__ERRNO: declare double @carg(double, double) [[READNONE]]
// NO__ERRNO: declare float @cargf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cargl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare double @carg(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @cargf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cargl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_casin(f);      __builtin_casinf(f);     __builtin_casinl(f);

// NO__ERRNO: declare { double, double } @casin(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casin(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_casinh(f);     __builtin_casinhf(f);    __builtin_casinhl(f); 

// NO__ERRNO: declare { double, double } @casinh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casinh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_catan(f);      __builtin_catanf(f);     __builtin_catanl(f); 

// NO__ERRNO: declare { double, double } @catan(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catan(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_catanh(f);     __builtin_catanhf(f);    __builtin_catanhl(f);

// NO__ERRNO: declare { double, double } @catanh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catanh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_ccos(f);       __builtin_ccosf(f);      __builtin_ccosl(f);

// NO__ERRNO: declare { double, double } @ccos(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccosf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccosl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccos(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccosf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccosl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_ccosh(f);      __builtin_ccoshf(f);     __builtin_ccoshl(f);

// NO__ERRNO: declare { double, double } @ccosh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccoshf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccosh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccoshf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_cexp(f);       __builtin_cexpf(f);      __builtin_cexpl(f);

// NO__ERRNO: declare { double, double } @cexp(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cexpf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cexpl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cexp(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cexpf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cexpl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

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

// NO__ERRNO: declare { double, double } @clog(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @clogf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @clogl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @clog(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @clogf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @clogl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_cproj(f);      __builtin_cprojf(f);     __builtin_cprojl(f); 

// NO__ERRNO: declare { double, double } @cproj(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cprojf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cprojl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cproj(double, double) [[READNONE:#[0-9]+]]
// HAS_ERRNO: declare <2 x float> @cprojf(<2 x float>) [[READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cprojl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_cpow(f,f);       __builtin_cpowf(f,f);      __builtin_cpowl(f,f);

// NO__ERRNO: declare { double, double } @cpow(double, double, double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cpowf(<2 x float>, <2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cpowl({ x86_fp80, x86_fp80 }* byval align 16, { x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cpow(double, double, double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cpowf(<2 x float>, <2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cpowl({ x86_fp80, x86_fp80 }* byval align 16, { x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_creal(f);      __builtin_crealf(f);     __builtin_creall(f);

// NO__ERRNO-NOT: .creal
// NO__ERRNO-NOT: @creal
// HAS_ERRNO-NOT: .creal
// HAS_ERRNO-NOT: @creal

  __builtin_csin(f);       __builtin_csinf(f);      __builtin_csinl(f);

// NO__ERRNO: declare { double, double } @csin(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csin(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_csinh(f);      __builtin_csinhf(f);     __builtin_csinhl(f);

// NO__ERRNO: declare { double, double } @csinh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csinh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_csqrt(f);      __builtin_csqrtf(f);     __builtin_csqrtl(f);  

// NO__ERRNO: declare { double, double } @csqrt(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csqrtf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csqrt(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csqrtf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_ctan(f);       __builtin_ctanf(f);      __builtin_ctanl(f);

// NO__ERRNO: declare { double, double } @ctan(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ctanf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ctanl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ctan(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ctanf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ctanl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]

  __builtin_ctanh(f);      __builtin_ctanhf(f);     __builtin_ctanhl(f); 

// NO__ERRNO: declare { double, double } @ctanh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ctanhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ctanhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ctanh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ctanhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ctanhl({ x86_fp80, x86_fp80 }* byval align 16) [[NOT_READNONE]]
};


// NO__ERRNO: attributes [[READNONE]] = { {{.*}}readnone{{.*}} }
// NO__ERRNO: attributes [[NOT_READNONE]] = { nounwind "correctly{{.*}} }

// HAS_ERRNO: attributes [[NOT_READNONE]] = { nounwind "correctly{{.*}} }
// HAS_ERRNO: attributes [[READNONE]] = { {{.*}}readnone{{.*}} }

