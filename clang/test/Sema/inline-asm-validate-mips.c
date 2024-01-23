// RUN: %clang_cc1 -triple mips -target-feature +soft-float -DSOFT_FLOAT_NO_CONSTRAINT_F -fsyntax-only -verify %s

#ifdef SOFT_FLOAT_NO_CONSTRAINT_F
void read_float(float p) {
    float result = p;
    __asm__("" ::"f"(result)); // expected-error{{invalid input constraint 'f' in asm}}
}
#endif // SOFT_FLOAT_NO_CONSTRAINT_F
