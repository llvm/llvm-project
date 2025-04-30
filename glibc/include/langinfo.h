#ifndef _LANGINFO_H

#include <locale/langinfo.h>

#ifndef _ISOMAC
libc_hidden_proto (nl_langinfo)

extern __typeof (nl_langinfo_l) __nl_langinfo_l;
libc_hidden_proto (__nl_langinfo_l)
#endif

#endif
