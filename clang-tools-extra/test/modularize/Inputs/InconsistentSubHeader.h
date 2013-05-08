// Set up so TypeInt only defined during InconsistentHeader1.h include.
#ifdef SYMBOL1
#define SYMBOL 1
#endif
#ifdef SYMBOL2
#define SYMBOL 2
#endif

#if SYMBOL == 1
typedef int TypeInt;
#endif
