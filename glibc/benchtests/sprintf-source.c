/* A set of arbitrarily selected positional format specifiers.  */
#define FORMAT1 "   %1$d: %2$c%3$c%4$c%5$c%6$c %7$20s %8$f (%9$02x)\n"
/* A matching, but arbitrarily selected, set of non-positional format specifiers.  */
#define FORMAT2 "   %d: %c%c%c%c%c %20s %f (%02x)\n"
/* Sufficiently large buffer.  */
char buf[256];
