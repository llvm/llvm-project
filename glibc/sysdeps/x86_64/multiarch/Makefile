ifeq ($(subdir),string)

sysdep_routines += strncat-c stpncpy-c strncpy-c \
		   strcmp-sse2 strcmp-sse2-unaligned strcmp-ssse3  \
		   strcmp-sse4_2 strcmp-avx2 \
		   strncmp-sse2 strncmp-ssse3 strncmp-sse4_2 strncmp-avx2 \
		   memchr-sse2 rawmemchr-sse2 memchr-avx2 rawmemchr-avx2 \
		   memrchr-sse2 memrchr-avx2 \
		   memcmp-sse2 \
		   memcmp-avx2-movbe \
		   memcmp-sse4 memcpy-ssse3 \
		   memmove-ssse3 \
		   memcpy-ssse3-back \
		   memmove-ssse3-back \
		   memmove-avx512-no-vzeroupper \
		   strcasecmp_l-sse2 strcasecmp_l-ssse3 \
		   strcasecmp_l-sse4_2 strcasecmp_l-avx \
		   strncase_l-sse2 strncase_l-ssse3 \
		   strncase_l-sse4_2 strncase_l-avx \
		   strchr-sse2 strchrnul-sse2 strchr-avx2 strchrnul-avx2 \
		   strrchr-sse2 strrchr-avx2 \
		   strlen-sse2 strnlen-sse2 strlen-avx2 strnlen-avx2 \
		   strcat-avx2 strncat-avx2 \
		   strcat-ssse3 strncat-ssse3\
		   strcpy-avx2 strncpy-avx2 \
		   strcpy-sse2 stpcpy-sse2 \
		   strcpy-ssse3 strncpy-ssse3 stpcpy-ssse3 stpncpy-ssse3 \
		   strcpy-sse2-unaligned strncpy-sse2-unaligned \
		   stpcpy-sse2-unaligned stpncpy-sse2-unaligned \
		   stpcpy-avx2 stpncpy-avx2 \
		   strcat-sse2 \
		   strcat-sse2-unaligned strncat-sse2-unaligned \
		   strchr-sse2-no-bsf memcmp-ssse3 strstr-sse2-unaligned \
		   strcspn-sse2 strpbrk-sse2 strspn-sse2 \
		   strcspn-c strpbrk-c strspn-c varshift \
		   memset-avx512-no-vzeroupper \
		   memmove-sse2-unaligned-erms \
		   memmove-avx-unaligned-erms \
		   memmove-avx512-unaligned-erms \
		   memset-sse2-unaligned-erms \
		   memset-avx2-unaligned-erms \
		   memset-avx512-unaligned-erms \
		   memchr-avx2-rtm \
		   memcmp-avx2-movbe-rtm \
		   memmove-avx-unaligned-erms-rtm \
		   memrchr-avx2-rtm \
		   memset-avx2-unaligned-erms-rtm \
		   rawmemchr-avx2-rtm \
		   strchr-avx2-rtm \
		   strcmp-avx2-rtm \
		   strchrnul-avx2-rtm \
		   stpcpy-avx2-rtm \
		   stpncpy-avx2-rtm \
		   strcat-avx2-rtm \
		   strcpy-avx2-rtm \
		   strlen-avx2-rtm \
		   strncat-avx2-rtm \
		   strncmp-avx2-rtm \
		   strncpy-avx2-rtm \
		   strnlen-avx2-rtm \
		   strrchr-avx2-rtm \
		   memchr-evex \
		   memcmp-evex-movbe \
		   memmove-evex-unaligned-erms \
		   memrchr-evex \
		   memset-evex-unaligned-erms \
		   rawmemchr-evex \
		   stpcpy-evex \
		   stpncpy-evex \
		   strcat-evex \
		   strchr-evex \
		   strchrnul-evex \
		   strcmp-evex \
		   strcpy-evex \
		   strlen-evex \
		   strncat-evex \
		   strncmp-evex \
		   strncpy-evex \
		   strnlen-evex \
		   strrchr-evex \
		   memchr-evex-rtm \
		   rawmemchr-evex-rtm
CFLAGS-varshift.c += -msse4
CFLAGS-strcspn-c.c += -msse4
CFLAGS-strpbrk-c.c += -msse4
CFLAGS-strspn-c.c += -msse4
endif

ifeq ($(subdir),wcsmbs)
sysdep_routines += wmemcmp-sse4 wmemcmp-ssse3 wmemcmp-c \
		   wmemcmp-avx2-movbe \
		   wmemchr-sse2 wmemchr-avx2 \
		   wcscmp-sse2 wcscmp-avx2 \
		   wcsncmp-sse2 wcsncmp-avx2 \
		   wcscpy-ssse3 wcscpy-c \
		   wcschr-sse2 wcschr-avx2 \
		   wcsrchr-sse2 wcsrchr-avx2 \
		   wcslen-sse2 wcslen-sse4_1 wcslen-avx2 \
		   wcsnlen-c wcsnlen-sse4_1 wcsnlen-avx2 \
		   wcschr-avx2-rtm \
		   wcscmp-avx2-rtm \
		   wcslen-avx2-rtm \
		   wcsncmp-avx2-rtm \
		   wcsnlen-avx2-rtm \
		   wcsrchr-avx2-rtm \
		   wmemchr-avx2-rtm \
		   wmemcmp-avx2-movbe-rtm \
		   wcschr-evex \
		   wcscmp-evex \
		   wcslen-evex \
		   wcsncmp-evex \
		   wcsnlen-evex \
		   wcsrchr-evex \
		   wmemchr-evex \
		   wmemcmp-evex-movbe \
		   wmemchr-evex-rtm
endif

ifeq ($(subdir),debug)
sysdep_routines += memcpy_chk-nonshared mempcpy_chk-nonshared \
		   memmove_chk-nonshared memset_chk-nonshared \
		   wmemset_chk-nonshared
endif
