#define FOO
#define BAR(X, Y) X, Y
#define IDENTITY(X) X
#define WIBBLE(...)
#define DEAD_MACRO
#undef DEAD_MACRO
#define MACRO_WITH_HISTORY a
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY b, c
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY(X, Y) X->Y
