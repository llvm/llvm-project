/* X32 uses 64-bit _itoa_word and _itoa is mapped to _itoa_word.  */
#define _ITOA_NEEDED		0
#define _ITOA_WORD_TYPE		unsigned long long int
#include_next <_itoa.h>
