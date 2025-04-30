#error "This file must be written based on the data type sizes of the target"

/* The following entries are a template for what defines should be in the
   wordsize.h header file for a target.  */

/* Size in bits of the 'long int' and pointer types.  */
#define __WORDSIZE

/* This should be set to 1 if __WORDSIZE is 32 and size_t is type
   'unsigned long' instead of type 'unsigned int'.  This will ensure
   that SIZE_MAX is defined as an unsigned long constant instead of an
   unsigned int constant.  Set to 0 if __WORDSIZE is 32 and size_t is
   'unsigned int' and leave undefined if __WORDSIZE is 64.  */
#define __WORDSIZE32_SIZE_ULONG

/* This should be set to 1 if __WORDSIZE is 32 and ptrdiff_t is type 'long'
   instead of type 'int'.  This will ensure that PTRDIFF_MIN and PTRDIFF_MAX
   are defined as long constants instead of int constants.  Set to 0 if
   __WORDSIZE is 32 and ptrdiff_t is type 'int' and leave undefined if
   __WORDSIZE is 64.  */
#define __WORDSIZE32_PTRDIFF_LONG

/* Set to 1 in order to force time types to be 32 bits instead of 64 bits in
   struct lastlog and struct utmp{,x} on 64-bit ports.  This may be done in
   order to make 64-bit ports compatible with 32-bit ports.  Set to 0 for
   64-bit ports where the time types are 64-bits or for any 32-bit ports.  */
#define __WORDSIZE_TIME64_COMPAT32
