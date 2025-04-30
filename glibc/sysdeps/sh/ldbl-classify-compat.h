#ifndef SH_LDBL_CLASSIFY_COMPAT_H
#define SH_LDBL_CLASSIFY_COMPAT_H 1

/* Enable __finitel, __isinfl, and __isnanl for binary compatibility
   when built without long double support. */
#define LDBL_CLASSIFY_COMPAT 1

#endif
