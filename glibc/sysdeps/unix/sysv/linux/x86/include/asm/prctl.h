/* FIXME: CET arch_prctl bits should come from the kernel header files.
   This file should be removed if <asm/prctl.h> from the required kernel
   header files contains CET arch_prctl bits.  */

#include_next <asm/prctl.h>

#ifndef ARCH_CET_STATUS
/* CET features:
   IBT:   GNU_PROPERTY_X86_FEATURE_1_IBT
   SHSTK: GNU_PROPERTY_X86_FEATURE_1_SHSTK
 */
/* Return CET features in unsigned long long *addr:
     features: addr[0].
     shadow stack base address: addr[1].
     shadow stack size: addr[2].
 */
# define ARCH_CET_STATUS	0x3001
/* Disable CET features in unsigned int features.  */
# define ARCH_CET_DISABLE	0x3002
/* Lock all CET features.  */
# define ARCH_CET_LOCK		0x3003
/* Allocate a new shadow stack with unsigned long long *addr:
     IN: requested shadow stack size: *addr.
     OUT: allocated shadow stack address: *addr.
 */
# define ARCH_CET_ALLOC_SHSTK	0x3004
#endif /* ARCH_CET_STATUS */
