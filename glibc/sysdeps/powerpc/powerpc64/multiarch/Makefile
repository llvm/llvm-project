ifeq ($(subdir),string)
sysdep_routines += memcpy-power8-cached memcpy-power7 memcpy-a2 memcpy-power6 \
		   memcpy-cell memcpy-power4 memcpy-ppc64 \
		   memcmp-power8 memcmp-power7 memcmp-power4 memcmp-ppc64 \
		   memset-power7 memset-power6 memset-power4 \
		   memset-ppc64 memset-power8 \
		   mempcpy-power7 mempcpy-ppc64 \
		   memchr-power8 memchr-power7 memchr-ppc64 \
		   memrchr-power8 memrchr-power7 memrchr-ppc64 \
		   rawmemchr-power7 rawmemchr-ppc64 \
		   strlen-power7 strlen-ppc64 \
		   strnlen-power8 strnlen-power7 strnlen-ppc64 \
		   strcasecmp-power7 strcasecmp_l-power7 \
		   strncase-power7 strncase_l-power7 \
		   strncmp-power8 strncmp-power7 \
		   strncmp-power4 strncmp-ppc64 \
		   strchr-power8 strchr-power7 strchr-ppc64 \
		   strchrnul-power8 strchrnul-power7 strchrnul-ppc64 \
		   strcpy-power8 strcpy-power7 strcpy-ppc64 stpcpy-power8 \
		   stpcpy-power7 stpcpy-ppc64 \
		   strrchr-power8 strrchr-power7 strrchr-ppc64 \
		   strncat-power8 strncat-power7 strncat-ppc64 \
		   strncpy-power7 strncpy-ppc64 \
		   stpncpy-power8 stpncpy-power7 stpncpy-ppc64 \
		   strcmp-power8 strcmp-power7 strcmp-ppc64 \
		   strcat-power8 strcat-power7 strcat-ppc64 \
		   memmove-power7 memmove-ppc64 wordcopy-ppc64 bcopy-ppc64 \
		   strncpy-power8 strstr-power7 strstr-ppc64 \
		   strspn-power8 strspn-ppc64 strcspn-power8 strcspn-ppc64 \
		   strlen-power8 strcasestr-power8 strcasestr-ppc64 \
		   strcasecmp-ppc64 strcasecmp-power8 strncase-ppc64 \
		   strncase-power8

ifneq (,$(filter %le,$(config-machine)))
sysdep_routines += memcmp-power10 memcpy-power10 memmove-power10 memset-power10 \
		   rawmemchr-power9 rawmemchr-power10 \
		   strcmp-power9 strncmp-power9 strcpy-power9 stpcpy-power9 \
		   strlen-power9 strncpy-power9 stpncpy-power9 strlen-power10
endif
CFLAGS-strncase-power7.c += -mcpu=power7 -funroll-loops
CFLAGS-strncase_l-power7.c += -mcpu=power7 -funroll-loops
endif
