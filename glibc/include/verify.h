/* Gnulib <verify.h>, simplified by assuming GCC 4.6 or later.  */
#define verify(R) _Static_assert (R, "verify (" #R ")")

#define assume(R) ((R) ? (void) 0 : __builtin_unreachable ())
