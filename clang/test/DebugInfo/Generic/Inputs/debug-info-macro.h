
#ifdef D1
/*Line 3*/ #define A(x, y, z) (x)
#endif

#ifdef D2
/*Line 7*/ #define A(x, y, z) (y)
#endif

#ifdef A
/*Line 11*/ #undef A
#endif
