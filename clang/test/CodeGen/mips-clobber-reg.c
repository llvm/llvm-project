// RUN: %clang -target mipsel-unknown-linux -S -o - -emit-llvm %s

/*
    This checks that the frontend will accept both
    enumerated and symbolic Mips register names.

    Includes:
    - GPR
    - FPU
    - MSA

    Any bad names will make the frontend choke.
 */

int main(void)
{

    __asm__ __volatile__ (".set noat \n\t addi $7,$at,77":::"at");
    __asm__ __volatile__ ("addi $7,$v0,77":::"v0");
    __asm__ __volatile__ ("addi $7,$v1,77":::"v1");
    __asm__ __volatile__ ("addi $7,$a0,77":::"a0");
    __asm__ __volatile__ ("addi $7,$a1,77":::"a1");
    __asm__ __volatile__ ("addi $7,$a2,77":::"a2");
    __asm__ __volatile__ ("addi $7,$a3,77":::"a3");
    __asm__ __volatile__ ("addi $7,$t0,77":::"t0");
    __asm__ __volatile__ ("addi $7,$t1,77":::"t1");
    __asm__ __volatile__ ("addi $7,$t2,77":::"t2");
    __asm__ __volatile__ ("addi $7,$t3,77":::"t3");
    __asm__ __volatile__ ("addi $7,$t4,77":::"t4");
    __asm__ __volatile__ ("addi $7,$t5,77":::"t5");
    __asm__ __volatile__ ("addi $7,$t6,77":::"t6");
    __asm__ __volatile__ ("addi $7,$t7,77":::"t7");
    __asm__ __volatile__ ("addi $7,$s0,77":::"s0");
    __asm__ __volatile__ ("addi $7,$s1,77":::"s1");
    __asm__ __volatile__ ("addi $7,$s2,77":::"s2");
    __asm__ __volatile__ ("addi $7,$s3,77":::"s3");
    __asm__ __volatile__ ("addi $7,$s4,77":::"s4");
    __asm__ __volatile__ ("addi $7,$s5,77":::"s5");
    __asm__ __volatile__ ("addi $7,$s6,77":::"s6");
    __asm__ __volatile__ ("addi $7,$s7,77":::"s7");
    __asm__ __volatile__ ("addi $7,$t8,77":::"t8");
    __asm__ __volatile__ ("addi $7,$t9,77":::"t9");
    __asm__ __volatile__ ("addi $7,$k0,77":::"k0");
    __asm__ __volatile__ ("addi $7,$k1,77":::"k1");
    __asm__ __volatile__ ("addi $7,$gp,77":::"gp");
    __asm__ __volatile__ ("addi $7,$sp,77":::"sp");
    __asm__ __volatile__ ("addi $7,$fp,77":::"fp");
    __asm__ __volatile__ ("addi $7,$sp,77":::"$sp");
    __asm__ __volatile__ ("addi $7,$fp,77":::"$fp");
    __asm__ __volatile__ ("addi $7,$ra,77":::"ra");

    __asm__ __volatile__ ("addi $7,$0,77":::"$0");
    __asm__ __volatile__ (".set noat \n\t addi $7,$1,77":::"$1");
    __asm__ __volatile__ ("addi $7,$2,77":::"$2");
    __asm__ __volatile__ ("addi $7,$3,77":::"$3");
    __asm__ __volatile__ ("addi $7,$4,77":::"$4");
    __asm__ __volatile__ ("addi $7,$5,77":::"$5");
    __asm__ __volatile__ ("addi $7,$6,77":::"$6");
    __asm__ __volatile__ ("addi $7,$7,77":::"$7");
    __asm__ __volatile__ ("addi $7,$8,77":::"$8");
    __asm__ __volatile__ ("addi $7,$9,77":::"$9");
    __asm__ __volatile__ ("addi $7,$10,77":::"$10");
    __asm__ __volatile__ ("addi $7,$11,77":::"$11");
    __asm__ __volatile__ ("addi $7,$12,77":::"$12");
    __asm__ __volatile__ ("addi $7,$13,77":::"$13");
    __asm__ __volatile__ ("addi $7,$14,77":::"$14");
    __asm__ __volatile__ ("addi $7,$15,77":::"$15");
    __asm__ __volatile__ ("addi $7,$16,77":::"$16");
    __asm__ __volatile__ ("addi $7,$17,77":::"$17");
    __asm__ __volatile__ ("addi $7,$18,77":::"$18");
    __asm__ __volatile__ ("addi $7,$19,77":::"$19");
    __asm__ __volatile__ ("addi $7,$20,77":::"$20");
    __asm__ __volatile__ ("addi $7,$21,77":::"$21");
    __asm__ __volatile__ ("addi $7,$22,77":::"$22");
    __asm__ __volatile__ ("addi $7,$23,77":::"$23");
    __asm__ __volatile__ ("addi $7,$24,77":::"$24");
    __asm__ __volatile__ ("addi $7,$25,77":::"$25");
    __asm__ __volatile__ ("addi $7,$26,77":::"$26");
    __asm__ __volatile__ ("addi $7,$27,77":::"$27");
    __asm__ __volatile__ ("addi $7,$28,77":::"$28");
    __asm__ __volatile__ ("addi $7,$29,77":::"$29");
    __asm__ __volatile__ ("addi $7,$30,77":::"$30");
    __asm__ __volatile__ ("addi $7,$31,77":::"$31");

    __asm__ __volatile__ ("fadd.s $f0,77":::"$f0");
    __asm__ __volatile__ ("fadd.s $f1,77":::"$f1");
    __asm__ __volatile__ ("fadd.s $f2,77":::"$f2");
    __asm__ __volatile__ ("fadd.s $f3,77":::"$f3");
    __asm__ __volatile__ ("fadd.s $f4,77":::"$f4");
    __asm__ __volatile__ ("fadd.s $f5,77":::"$f5");
    __asm__ __volatile__ ("fadd.s $f6,77":::"$f6");
    __asm__ __volatile__ ("fadd.s $f7,77":::"$f7");
    __asm__ __volatile__ ("fadd.s $f8,77":::"$f8");
    __asm__ __volatile__ ("fadd.s $f9,77":::"$f9");
    __asm__ __volatile__ ("fadd.s $f10,77":::"$f10");
    __asm__ __volatile__ ("fadd.s $f11,77":::"$f11");
    __asm__ __volatile__ ("fadd.s $f12,77":::"$f12");
    __asm__ __volatile__ ("fadd.s $f13,77":::"$f13");
    __asm__ __volatile__ ("fadd.s $f14,77":::"$f14");
    __asm__ __volatile__ ("fadd.s $f15,77":::"$f15");
    __asm__ __volatile__ ("fadd.s $f16,77":::"$f16");
    __asm__ __volatile__ ("fadd.s $f17,77":::"$f17");
    __asm__ __volatile__ ("fadd.s $f18,77":::"$f18");
    __asm__ __volatile__ ("fadd.s $f19,77":::"$f19");
    __asm__ __volatile__ ("fadd.s $f20,77":::"$f20");
    __asm__ __volatile__ ("fadd.s $f21,77":::"$f21");
    __asm__ __volatile__ ("fadd.s $f22,77":::"$f22");
    __asm__ __volatile__ ("fadd.s $f23,77":::"$f23");
    __asm__ __volatile__ ("fadd.s $f24,77":::"$f24");
    __asm__ __volatile__ ("fadd.s $f25,77":::"$f25");
    __asm__ __volatile__ ("fadd.s $f26,77":::"$f26");
    __asm__ __volatile__ ("fadd.s $f27,77":::"$f27");
    __asm__ __volatile__ ("fadd.s $f28,77":::"$f28");
    __asm__ __volatile__ ("fadd.s $f29,77":::"$f29");
    __asm__ __volatile__ ("fadd.s $f30,77":::"$f30");
    __asm__ __volatile__ ("fadd.s $f31,77":::"$f31");

    __asm__ __volatile__ ("ldi.w $w0,77":::"$w0");
    __asm__ __volatile__ ("ldi.w $w1,77":::"$w1");
    __asm__ __volatile__ ("ldi.w $w2,77":::"$w2");
    __asm__ __volatile__ ("ldi.w $w3,77":::"$w3");
    __asm__ __volatile__ ("ldi.w $w4,77":::"$w4");
    __asm__ __volatile__ ("ldi.w $w5,77":::"$w5");
    __asm__ __volatile__ ("ldi.w $w6,77":::"$w6");
    __asm__ __volatile__ ("ldi.w $w7,77":::"$w7");
    __asm__ __volatile__ ("ldi.w $w8,77":::"$w8");
    __asm__ __volatile__ ("ldi.w $w9,77":::"$w9");
    __asm__ __volatile__ ("ldi.w $w10,77":::"$w10");
    __asm__ __volatile__ ("ldi.w $w11,77":::"$w10");
    __asm__ __volatile__ ("ldi.w $w12,77":::"$w12");
    __asm__ __volatile__ ("ldi.w $w13,77":::"$w13");
    __asm__ __volatile__ ("ldi.w $w14,77":::"$w14");
    __asm__ __volatile__ ("ldi.w $w15,77":::"$w15");
    __asm__ __volatile__ ("ldi.w $w16,77":::"$w16");
    __asm__ __volatile__ ("ldi.w $w17,77":::"$w17");
    __asm__ __volatile__ ("ldi.w $w18,77":::"$w18");
    __asm__ __volatile__ ("ldi.w $w19,77":::"$w19");
    __asm__ __volatile__ ("ldi.w $w20,77":::"$w20");
    __asm__ __volatile__ ("ldi.w $w21,77":::"$w21");
    __asm__ __volatile__ ("ldi.w $w22,77":::"$w22");
    __asm__ __volatile__ ("ldi.w $w23,77":::"$w23");
    __asm__ __volatile__ ("ldi.w $w24,77":::"$w24");
    __asm__ __volatile__ ("ldi.w $w25,77":::"$w25");
    __asm__ __volatile__ ("ldi.w $w26,77":::"$w26");
    __asm__ __volatile__ ("ldi.w $w27,77":::"$w27");
    __asm__ __volatile__ ("ldi.w $w28,77":::"$w28");
    __asm__ __volatile__ ("ldi.w $w29,77":::"$w29");
    __asm__ __volatile__ ("ldi.w $w30,77":::"$w30");
    __asm__ __volatile__ ("ldi.w $w31,77":::"$w31");
}
