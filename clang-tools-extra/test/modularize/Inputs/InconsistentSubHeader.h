// Set up so TypeInt only defined during InconsistentHeader1.h include.
#ifdef SYMBOL1
#define SYMBOL 1
#define FUNC_STYLE(a, b) a||b
#endif
#ifdef SYMBOL2
#define SYMBOL 2
#define FUNC_STYLE(a, b) a&&b
#endif

#if SYMBOL == 1
typedef int TypeInt;
#endif

int var = FUNC_STYLE(1, 0);

#if defined(SYMBOL1)
#endif
