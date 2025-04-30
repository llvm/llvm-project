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

ifeq ($(subdir),stdlib)
sysdep_routines += mul_1-vis3 mul_1-generic addmul_1-vis3 addmul_1-generic \
		   submul_1-vis3 submul_1-generic add_n-vis3 add_n-generic \
		   sub_n-vis3 sub_n-generic
endif

ifeq ($(subdir),math)
gmp-sysdep_routines = mul_1-vis3 addmul_1-vis3 submul_1-vis3 add_n-vis3 \
		      sub_n-vis3
endif
