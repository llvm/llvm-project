/* Used by the elf ifunc tests.  */
#ifndef ELF_IFUNC_SEL_H
#define ELF_IFUNC_SEL_H 1

extern int global;

static inline __attribute__ ((always_inline)) void *
inhibit_stack_protector
ifunc_sel (int (*f1) (void), int (*f2) (void), int (*f3) (void))
{
  register void *ret __asm__ ("r3");
  __asm__ ("mflr 12\n\t"
	   "bcl 20,31,1f\n"
	   "1:\tmflr 11\n\t"
	   "mtlr 12\n\t"
	   "addis 12,11,global-1b@ha\n\t"
	   "lwz 12,global-1b@l(12)\n\t"
	   "addis %0,11,%2-1b@ha\n\t"
	   "addi %0,%0,%2-1b@l\n\t"
	   "cmpwi 12,1\n\t"
	   "beq 2f\n\t"
	   "addis %0,11,%3-1b@ha\n\t"
	   "addi %0,%0,%3-1b@l\n\t"
	   "cmpwi 12,-1\n\t"
	   "beq 2f\n\t"
	   "addis %0,11,%4-1b@ha\n\t"
	   "addi %0,%0,%4-1b@l\n\t"
	   "2:"
	   : "=r" (ret)
	   : "i" (&global), "i" (f1), "i" (f2), "i" (f3)
	   : "11", "12", "cr0");
  return ret;
}

static inline __attribute__ ((always_inline)) void *
inhibit_stack_protector
ifunc_one (int (*f1) (void))
{
  register void *ret __asm__ ("r3");
  __asm__ ("mflr 12\n\t"
	   "bcl 20,31,1f\n"
	   "1:\tmflr %0\n\t"
	   "mtlr 12\n\t"
	   "addis %0,%0,%1-1b@ha\n\t"
	   "addi %0,%0,%1-1b@l"
	   : "=r" (ret)
	   : "i" (f1)
	   : "12");
  return ret;
}
#endif
