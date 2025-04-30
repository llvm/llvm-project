/* PPCA2 has a different cache-line size than the usual 128 bytes.  To avoid
   using code that assumes cache-line size to be 128 bytes (with dcbz
   instructions) we use the generic code instead.  */
#include <string/memset.c>
