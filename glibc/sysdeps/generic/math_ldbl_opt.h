/* -mlong-double-64 compatibility mode macros.  Stub version.

   These macros are used by some math/ and sysdeps/ieee754/ code.
   These are the generic definitions for when no funny business is going on.
   sysdeps/ieee754/ldbl-opt/math_ldbl_opt.h defines them differently
   for platforms where compatibility symbols are required for a previous
   ABI that defined long double functions as aliases for the double code.  */

#include <shlib-compat.h>

#define LONG_DOUBLE_COMPAT(lib, introduced) 0
#define long_double_symbol(lib, local, symbol)
#define ldbl_hidden_def(local, name) libc_hidden_def (name)
#define ldbl_strong_alias(name, aliasname) strong_alias (name, aliasname)
#define ldbl_weak_alias(name, aliasname) weak_alias (name, aliasname)
#define ldbl_compat_symbol(lib, local, symbol, version) \
  compat_symbol (lib, local, symbol, version)
