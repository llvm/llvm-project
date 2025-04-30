ifeq ($(subdir),string)
gen-as-const-headers += locale-defines.sym
sysdep_routines += bzero-sse2 memset-sse2 memcpy-ssse3 mempcpy-ssse3 \
		   memmove-ssse3 memcpy-ssse3-rep mempcpy-ssse3-rep \
		   memmove-ssse3-rep bcopy-ssse3 bcopy-ssse3-rep \
		   memset-sse2-rep bzero-sse2-rep strcmp-ssse3 \
		   strcmp-sse4 strncmp-c strncmp-ssse3 strncmp-sse4 \
		   memcmp-ssse3 memcmp-sse4 varshift \
		   strlen-sse2 strlen-sse2-bsf strncpy-c strcpy-ssse3 \
		   strncpy-ssse3 stpcpy-ssse3 stpncpy-ssse3 strcpy-sse2 \
		   strncpy-sse2 stpcpy-sse2 stpncpy-sse2 strcat-ssse3 \
		   strcat-sse2 strncat-ssse3 strncat-sse2 strncat-c \
		   strchr-sse2 strrchr-sse2 strchr-sse2-bsf strrchr-sse2-bsf \
		   memchr-sse2 memchr-sse2-bsf \
		   memrchr-sse2 memrchr-sse2-bsf memrchr-c \
		   rawmemchr-sse2 rawmemchr-sse2-bsf \
		   strnlen-sse2 strnlen-c \
		   strcasecmp_l-c strcasecmp-c strcasecmp_l-ssse3 \
		   strncase_l-c strncase-c strncase_l-ssse3 \
		   strcasecmp_l-sse4 strncase_l-sse4 \
		   bcopy-sse2-unaligned memcpy-sse2-unaligned \
		   mempcpy-sse2-unaligned memmove-sse2-unaligned \
		   strcspn-c strpbrk-c strspn-c \
		   bcopy-ia32 bzero-ia32 rawmemchr-ia32 \
		   memchr-ia32 memcmp-ia32 memcpy-ia32 memmove-ia32 \
		   mempcpy-ia32 memset-ia32 strcat-ia32 strchr-ia32 \
		   strrchr-ia32 strcpy-ia32 strcmp-ia32 strcspn-ia32 \
		   strpbrk-ia32 strspn-ia32 strlen-ia32 stpcpy-ia32 \
		   stpncpy-ia32
CFLAGS-varshift.c += -msse4
CFLAGS-strcspn-c.c += -msse4
CFLAGS-strpbrk-c.c += -msse4
CFLAGS-strspn-c.c += -msse4
endif

ifeq ($(subdir),wcsmbs)
sysdep_routines += wcscmp-sse2 wcscmp-c wcslen-sse2 wcslen-c \
		   wmemcmp-sse4 wmemcmp-ssse3 wmemcmp-c wcschr-sse2 \
		   wcschr-c wcsrchr-sse2 wcsrchr-c wcscpy-ssse3 wcscpy-c
endif

ifeq ($(subdir),math)
libm-sysdep_routines += s_fma-fma s_fmaf-fma
CFLAGS-s_fma-fma.c += -mavx -mfpmath=sse
CFLAGS-s_fmaf-fma.c += -mavx -mfpmath=sse
endif

ifeq ($(subdir),debug)
sysdep_routines += memcpy_chk-nonshared mempcpy_chk-nonshared \
		   memmove_chk-nonshared memset_chk-nonshared
endif
