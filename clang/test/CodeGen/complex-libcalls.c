// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -S -o - -emit-llvm              %s | FileCheck %s -check-prefix=NO__ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -S -o - -emit-llvm -fmath-errno %s | FileCheck %s -check-prefix=HAS_ERRNO

// Test attributes and builtin codegen of complex library calls. 

void foo(float f) {
  cabs(f);       cabsf(f);      cabsl(f);

// NO__ERRNO: declare double @cabs(double, double) [[READNONE:#[0-9]+]]
// NO__ERRNO: declare float @cabsf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cabsl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare double @cabs(double, double) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @cabsf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cabsl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cacos(f);      cacosf(f);     cacosl(f);

// NO__ERRNO: declare { double, double } @cacos(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacosf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacosl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacos(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacosf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacosl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cacosh(f);     cacoshf(f);    cacoshl(f);

// NO__ERRNO: declare { double, double } @cacosh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cacoshf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cacosh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cacoshf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cacoshl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  carg(f);       cargf(f);      cargl(f);

// NO__ERRNO: declare double @carg(double, double) [[READNONE]]
// NO__ERRNO: declare float @cargf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cargl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare double @carg(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @cargf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cargl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  casin(f);      casinf(f);     casinl(f);

// NO__ERRNO: declare { double, double } @casin(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casin(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  casinh(f);     casinhf(f);    casinhl(f); 

// NO__ERRNO: declare { double, double } @casinh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @casinhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @casinhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @casinh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @casinhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @casinhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  catan(f);      catanf(f);     catanl(f); 

// NO__ERRNO: declare { double, double } @catan(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catan(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  catanh(f);     catanhf(f);    catanhl(f);

// NO__ERRNO: declare { double, double } @catanh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @catanhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @catanhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @catanh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @catanhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @catanhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ccos(f);       ccosf(f);      ccosl(f);

// NO__ERRNO: declare { double, double } @ccos(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccosf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccosl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccos(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccosf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccosl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ccosh(f);      ccoshf(f);     ccoshl(f);

// NO__ERRNO: declare { double, double } @ccosh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ccoshf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ccosh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ccoshf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ccoshl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cexp(f);       cexpf(f);      cexpl(f);

// NO__ERRNO: declare { double, double } @cexp(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cexpf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cexpl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cexp(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cexpf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cexpl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

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

// NO__ERRNO: declare { double, double } @clog(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @clogf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @clogl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @clog(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @clogf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @clogl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cproj(f);      cprojf(f);     cprojl(f); 

// NO__ERRNO: declare { double, double } @cproj(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cprojf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cprojl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cproj(double, double) [[READNONE:#[0-9]+]]
// HAS_ERRNO: declare <2 x float> @cprojf(<2 x float>) [[READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cprojl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  cpow(f,f);       cpowf(f,f);      cpowl(f,f);

// NO__ERRNO: declare { double, double } @cpow(double, double, double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @cpowf(<2 x float>, <2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @cpowl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16, { x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @cpow(double, double, double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @cpowf(<2 x float>, <2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @cpowl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16, { x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  creal(f);      crealf(f);     creall(f);

// NO__ERRNO-NOT: .creal
// NO__ERRNO-NOT: @creal
// HAS_ERRNO-NOT: .creal
// HAS_ERRNO-NOT: @creal

  csin(f);       csinf(f);      csinl(f);

// NO__ERRNO: declare { double, double } @csin(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csin(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  csinh(f);      csinhf(f);     csinhl(f);

// NO__ERRNO: declare { double, double } @csinh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csinhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csinhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csinh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csinhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csinhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  csqrt(f);      csqrtf(f);     csqrtl(f);  

// NO__ERRNO: declare { double, double } @csqrt(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @csqrtf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @csqrt(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @csqrtf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @csqrtl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ctan(f);       ctanf(f);      ctanl(f);

// NO__ERRNO: declare { double, double } @ctan(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ctanf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ctanl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ctan(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ctanf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ctanl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]

  ctanh(f);      ctanhf(f);     ctanhl(f); 

// NO__ERRNO: declare { double, double } @ctanh(double, double) [[READNONE]]
// NO__ERRNO: declare <2 x float> @ctanhf(<2 x float>) [[READNONE]]
// NO__ERRNO: declare { x86_fp80, x86_fp80 } @ctanhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
// HAS_ERRNO: declare { double, double } @ctanh(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare <2 x float> @ctanhf(<2 x float>) [[NOT_READNONE]]
// HAS_ERRNO: declare { x86_fp80, x86_fp80 } @ctanhl({ x86_fp80, x86_fp80 }* byval({ x86_fp80, x86_fp80 }) align 16) [[NOT_READNONE]]
};

// NO__ERRNO: attributes [[READNONE]] = { {{.*}}readnone{{.*}} }
// NO__ERRNO: attributes [[NOT_READNONE]] = { nounwind {{.*}} }

// HAS_ERRNO: attributes [[NOT_READNONE]] = { nounwind {{.*}} }
// HAS_ERRNO: attributes [[READNONE]] = { {{.*}}readnone{{.*}} }
