/* shared library to test for __tls_get_addr optimization.  */
#include <stdio.h>

#include "../../elf/tls-macros.h"
#include "dl-tls.h"

/* common 'int' variable in TLS.  */
COMMON_INT_DEF(foo);


int
tls_get_addr_opt_test (void)
{
  int result = 0;

  /* Get variable using general dynamic model.  */
  int *ap = TLS_GD (foo);
  if (*ap != 0)
    {
      printf ("foo = %d\n", *ap);
      result = 1;
    }

  tls_index *tls_arg;
#ifdef __powerpc64__
  register unsigned long thread_pointer __asm__ ("r13");
  asm ("addi %0,2,foo@got@tlsgd" : "=r" (tls_arg));
#else
  register unsigned long thread_pointer __asm__ ("r2");
  asm ("bcl 20,31,1f\n1:\t"
       "mflr %0\n\t"
       "addis %0,%0,_GLOBAL_OFFSET_TABLE_-1b@ha\n\t"
       "addi %0,%0,_GLOBAL_OFFSET_TABLE_-1b@l\n\t"
       "addi %0,%0,foo@got@tlsgd" : "=b" (tls_arg));
#endif

  if (tls_arg->ti_module != 0)
    {
      printf ("tls_index not optimized, binutils too old?\n");
      result = 1;
    }
  else if (tls_arg->ti_offset + thread_pointer != (unsigned long) ap)
    {
      printf ("tls_index->ti_offset wrong value\n");
      result = 1;
    }

  return result;
}
