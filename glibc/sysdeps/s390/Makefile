ifeq ($(subdir),iconvdata)
ISO-8859-1_CP037_Z900-routines := iso-8859-1_cp037_z900
ISO-8859-1_CP037_Z900-map := gconv.map

UTF8_UTF32_Z9-routines := utf8-utf32-z9
UTF8_UTF32_Z9-map := gconv.map

UTF16_UTF32_Z9-routines := utf16-utf32-z9
UTF16_UTF32_Z9-map := gconv.map

UTF8_UTF16_Z9-routines := utf8-utf16-z9
UTF8_UTF16_Z9-map := gconv.map

s390x-iconv-modules = ISO-8859-1_CP037_Z900 UTF8_UTF16_Z9 UTF16_UTF32_Z9 UTF8_UTF32_Z9

extra-modules-left += $(s390x-iconv-modules)
include extra-module.mk

cpp-srcs-left := $(foreach mod,$(s390x-iconv-modules),$($(mod)-routines))
lib := iconvdata
include $(patsubst %,$(..)libof-iterator.mk,$(cpp-srcs-left))

extra-objs      += $(addsuffix .so, $(s390x-iconv-modules))
install-others  += $(patsubst %, $(inst_gconvdir)/%.so, \
				 $(s390x-iconv-modules)) \
		   $(inst_gconvdir)/gconv-modules.d/gconv-modules-s390.conf

$(patsubst %, $(inst_gconvdir)/%.so, $(s390x-iconv-modules)) : \
$(inst_gconvdir)/%.so: $(objpfx)%.so $(+force)
	$(do-install-program)

ifdef objpfx
generated += gconv-modules.d/gconv-modules-s390.conf
endif

$(inst_gconvdir)/gconv-modules.d/gconv-modules-s390.conf: \
		$(modpfx)gconv-modules-s390.conf $(+force)
	$(do-install)

$(modpfx)gconv-modules-s390.conf: ../sysdeps/s390/gconv-modules-s390.conf \
				  $(modpfx)
	cp $< $@
endif

ifeq ($(subdir),elf)
ifeq ($(build-shared),yes)
tests += tst-dl-runtime-resolve-noaudit tst-dl-runtime-resolve-audit \
	tst-dl-runtime-profile-noaudit
# FIXME: When both LD_PROFILE and LD_AUDIT are set, a segfault occurs.
#tests += tst-dl-runtime-profile-audit
modules-names += tst-dl-runtime-mod
$(objpfx)tst-dl-runtime-resolve-noaudit: $(objpfx)tst-dl-runtime-mod.so
$(objpfx)tst-dl-runtime-resolve-audit: $(objpfx)tst-dl-runtime-mod.so
$(objpfx)tst-dl-runtime-profile-noaudit: $(objpfx)tst-dl-runtime-mod.so
$(objpfx)tst-dl-runtime-profile-audit: $(objpfx)tst-dl-runtime-mod.so
$(objpfx)tst-dl-runtime-resolve-audit.out: $(objpfx)tst-auditmod1.so
$(objpfx)tst-dl-runtime-profile-audit.out: $(objpfx)tst-auditmod1.so
# The profiling output goes to $LD_PROFILE_OUTPUT/$LD_PROFILE.profile
env-profile = LD_PROFILE=$(objpfx)tst-dl-runtime-mod.so \
	LD_PROFILE_OUTPUT=/
env-audit = LD_AUDIT=$(objpfx)tst-auditmod1.so
tst-dl-runtime-resolve-audit-ENV = $(env-audit)
tst-dl-runtime-profile-noaudit-ENV = $(env-profile)
tst-dl-runtime-profile-audit-ENV = $(env-profile) $(env-audit)
endif
endif

ifeq ($(subdir),string)
sysdep_routines += bzero memset memset-z900 \
		   memcmp memcmp-z900 \
		   mempcpy memcpy memcpy-z900 \
		   memmove memmove-c \
		   strstr strstr-arch13 strstr-vx strstr-c \
		   memmem memmem-arch13 memmem-vx memmem-c \
		   strlen strlen-vx strlen-c \
		   strnlen strnlen-vx strnlen-c \
		   strcpy strcpy-vx strcpy-z900 \
		   stpcpy stpcpy-vx stpcpy-c \
		   strncpy strncpy-vx strncpy-z900 \
		   stpncpy stpncpy-vx stpncpy-c \
		   strcat strcat-vx strcat-c \
		   strncat strncat-vx strncat-c \
		   strcmp strcmp-vx strcmp-z900 \
		   strncmp strncmp-vx strncmp-c \
		   strchr strchr-vx strchr-c \
		   strchrnul strchrnul-vx strchrnul-c \
		   strrchr strrchr-vx strrchr-c \
		   strspn strspn-vx strspn-c \
		   strpbrk strpbrk-vx strpbrk-c \
		   strcspn strcspn-vx strcspn-c \
		   memchr memchr-vx memchr-z900 \
		   rawmemchr rawmemchr-vx rawmemchr-c \
		   memccpy memccpy-vx memccpy-c \
		   memrchr memrchr-vx memrchr-c
endif

ifeq ($(subdir),wcsmbs)
sysdep_routines += wcslen wcslen-vx wcslen-c \
		   wcsnlen wcsnlen-vx wcsnlen-c \
		   wcscpy wcscpy-vx wcscpy-c \
		   wcpcpy wcpcpy-vx wcpcpy-c \
		   wcsncpy wcsncpy-vx wcsncpy-c \
		   wcpncpy wcpncpy-vx wcpncpy-c \
		   wcscat wcscat-vx wcscat-c \
		   wcsncat wcsncat-vx wcsncat-c \
		   wcscmp wcscmp-vx wcscmp-c \
		   wcsncmp wcsncmp-vx wcsncmp-c \
		   wcschr wcschr-vx wcschr-c \
		   wcschrnul wcschrnul-vx wcschrnul-c \
		   wcsrchr wcsrchr-vx wcsrchr-c \
		   wcsspn wcsspn-vx wcsspn-c \
		   wcspbrk wcspbrk-vx wcspbrk-c \
		   wcscspn wcscspn-vx wcscspn-c \
		   wmemchr wmemchr-vx wmemchr-c \
		   wmemset wmemset-vx wmemset-c \
		   wmemcmp wmemcmp-vx wmemcmp-c
endif
