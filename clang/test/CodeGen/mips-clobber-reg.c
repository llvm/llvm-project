// RUN: %clang -target mipsel-unknown-linux -S -o - -emit-llvm %s 

/*
    This checks that the frontend will accept both
    enumerated and symbolic Mips GPR register names.
    
    Any bad names will make the frontend choke.
 */

main()
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
    __asm__ __volatile__ ("addi $7,$11,77":::"$10"); 
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

}
