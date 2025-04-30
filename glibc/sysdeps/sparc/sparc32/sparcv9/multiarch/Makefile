ifeq ($(subdir),crypt)
libcrypt-sysdep_routines += md5-crop sha256-crop sha512-crop
endif

ifeq ($(subdir),locale)
localedef-aux += md5-crop
endif

ifeq ($(subdir),string)
sysdep_routines += memcpy-ultra3 memcpy-niagara1 memcpy-niagara2 \
		   memset-niagara1 memcpy-niagara4 memset-niagara4 \
		   memcpy-ultra1 memset-ultra1 memcpy-memmove-niagara7 \
		   memmove-ultra1 memset-niagara7
endif
