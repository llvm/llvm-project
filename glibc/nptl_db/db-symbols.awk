# This script processes the libc.so abilist (with GLIBC_PRIVATE
# symbols included).  It checks for all the symbols used in td_symbol_list.

BEGIN {
%define DB_MAIN_VARIABLE(name) /* Nothing. */
%define DB_MAIN_SYMBOL(name) /* Nothing. */
%define DB_MAIN_ARRAY_VARIABLE(name) /* Nothing. */
%define DB_LOOKUP_NAME(idx, name)		required[STRINGIFY (name)] = 1;
%define DB_LOOKUP_NAME_TH_UNIQUE(idx, name)	th_unique[STRINGIFY (name)] = 1;
%include "db-symbols.h"

   in_symtab = 0;
}

/^GLIBC_PRIVATE / {
    seen[$2] = 1
}

END {
  status = 0;

  for (s in required) {
    if (s in seen) print s, "ok";
    else {
      status = 1;
      print s, "***MISSING***";
    }
  }

  any = "";
  for (s in th_unique) {
    if (s in seen) {
      any = s;
      break;
    }
  }
  if (any)
    print "th_unique:", any;
  else {
    status = 1;
    print "th_unique:", "***MISSING***";
  }

  exit(status);
}
