/* Used by the elf ifunc tests.  */
#ifndef ELF_IFUNC_SEL_H
#define ELF_IFUNC_SEL_H 1

extern int global;

static inline void *
inhibit_stack_protector
ifunc_sel (int (*f1) (void), int (*f2) (void), int (*f3) (void))
{
 switch (global)
   {
   case 1:
     return f1;
   case -1:
     return f2;
   default:
     return f3;
   }
}

static inline void *
inhibit_stack_protector
ifunc_one (int (*f1) (void))
{
  return f1;
}
#endif
