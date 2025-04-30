#ifndef _GSHADOW_H
#include <gshadow/gshadow.h>

# ifndef _ISOMAC

extern int __fgetsgent_r (FILE *stream, struct sgrp *resbuf, char *buffer,
			  size_t buflen, struct sgrp **result)
     attribute_hidden;
extern int __sgetsgent_r (const char *string, struct sgrp *resbuf,
			  char *buffer, size_t buflen, struct sgrp **result)
     attribute_hidden;

# endif /* !_ISOMAC */
#endif
