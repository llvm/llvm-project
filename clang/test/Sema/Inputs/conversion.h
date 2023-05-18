/* Fake system header for Sema/conversion.c */

#define LONG_MAX __LONG_MAX__
#define SETBIT(set,bit) do { int i = bit; set[i/(8*sizeof(set[0]))] |= (1 << (i%(8*sizeof(set)))); } while(0)
