# The`long double' type is a distinct type we support if
# -mlong-double-128 option is used (or when it becomes a default
# when -mlong-double-64 is not used).
long-double-fcts = yes
sysdep-CFLAGS += -mlong-double-128

ifeq ($(subdir),stdlib)
tests += tst-strtold-ldbl-128ibm
$(objpfx)tst-strtold-ldbl-128ibm: $(libm)
endif

ldbl-tests = test-fmodl-ldbl-128ibm test-remainderl-ldbl-128ibm \
	     test-remquol-ldbl-128ibm test-canonical-ldbl-128ibm \
	     test-totalorderl-ldbl-128ibm

ifeq ($(subdir),math)
tests += $(ldbl-tests)
endif

# Long double files may need extra CFLAGS.
ldbl-128ibm-routines = s_nexttoward s_nexttowardf \
		       $(type-ldouble-routines) \
		       $(subst F,$(type-ldouble-suffix),$(libm-compat-calls)) \
		       $(subst F,$(type-ldouble-suffix),$(libm-calls)) \
		       $(subst F,$(type-ldouble-suffix),$(calls)) \
		       $(foreach f,$(libm-narrow-fns), \
				 $(subst F,$(f), \
					 $(libm-narrow-types-ldouble-yes)))
$(foreach suf,$(all-object-suffixes), \
	  $(foreach r,$(ldbl-128ibm-routines) $(ldbl-tests), \
		    $(objpfx)$(r)$(suf))): \
  CFLAGS += $(type-ldouble-CFLAGS)
