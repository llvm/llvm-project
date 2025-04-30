/* Define a constant for the dgettext domainname for libc internal messages,
   so the string constant is not repeated in dozens of object files.  */

#include <libintl.h>

const char _libc_intl_domainname[] = "libc";
libc_hidden_data_def (_libc_intl_domainname)
