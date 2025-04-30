ifeq ($(subdir),string)
sysdep_routines += memcpy-power7 memcpy-a2 memcpy-power6 memcpy-cell \
		   memcpy-ppc32 memcmp-power7 memcmp-ppc32 memset-power7 \
		   memset-power6 memset-ppc32 bzero-power7 bzero-power6 \
		   bzero-ppc32 mempcpy-power7 mempcpy-ppc32 memchr-power7 \
		   memchr-ppc32 memrchr-power7 memrchr-ppc32 rawmemchr-power7 \
		   rawmemchr-ppc32 strlen-power7 strlen-ppc32 strnlen-power7 \
		   strnlen-ppc32 strncmp-power7 strncmp-ppc32 \
		   strcasecmp-power7 strcasecmp_l-power7 strncase-power7 \
		   strncase_l-power7 strchrnul-power7 strchrnul-ppc32 \
		   strchr-power7 strchr-ppc32 \
		   wordcopy-power7 wordcopy-ppc32 \
		   memmove-power7 memmove-ppc

CFLAGS-strncase-power7.c += -mcpu=power7 -funroll-loops
CFLAGS-strncase_l-power7.c += -mcpu=power7 -funroll-loops
endif
