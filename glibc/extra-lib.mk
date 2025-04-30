# This file is included several times in a row, once
# for each element of $(extra-libs).  $(extra-libs-left)
# is initialized first to $(extra-libs) so that with each
# inclusion, we advance $(lib) to the next library name (e.g. libfoo).
# The variable $($(lib)-routines) defines the list of modules
# to be included in that library.  A sysdep Makefile can add to
# $(lib)-sysdep_routines to include additional modules.
#
# Libraries listed in $(extra-libs-noinstall) are built, but not
# installed.

lib := $(firstword $(extra-libs-left))
extra-libs-left := $(filter-out $(lib),$(extra-libs-left))

object-suffixes-$(lib) := $(filter-out $($(lib)-inhibit-o),$(object-suffixes))

ifneq (,$($(lib)-static-only-routines))
ifneq (,$(filter yes%,$(build-shared)$($(lib).so-version)))
object-suffixes-$(lib) += $(filter-out $($(lib)-inhibit-o),.oS)
endif
endif

ifneq (,$(object-suffixes-$(lib)))

# Make sure these are simply-expanded variables before we append to them,
# since we want the expressions we append to be expanded right now.
install-lib := $(install-lib)
extra-objs := $(extra-objs)

# The modules that go in $(lib).
all-$(lib)-routines := $($(lib)-routines) $($(lib)-sysdep_routines)

# Add each flavor of library to the lists of things to build and install.
ifeq (,$(filter $(lib), $(extra-libs-noinstall)))
install-lib += $(foreach o,$(object-suffixes-$(lib)),$(lib:lib%=$(libtype$o)))
endif
extra-objs += $(foreach o,$(filter-out .os .oS,$(object-suffixes-$(lib))),\
			$(patsubst %,%$o,$(filter-out \
					   $($(lib)-shared-only-routines),\
					   $(all-$(lib)-routines))))
ifneq (,$(filter .os,$(object-suffixes-$(lib))))
extra-objs += $(patsubst %,%.os,$(filter-out $($(lib)-static-only-routines),\
					     $(all-$(lib)-routines)))
endif
ifneq (,$(filter .oS,$(object-suffixes-$(lib))))
extra-objs += $(patsubst %,%.oS,$(filter $($(lib)-static-only-routines),\
					 $(all-$(lib)-routines)))
endif
alltypes-$(lib) := $(foreach o,$(object-suffixes-$(lib)),\
			     $(objpfx)$(patsubst %,$(libtype$o),\
			     $(lib:lib%=%)))

ifeq (,$(filter $(lib),$(extra-libs-others)))
lib-noranlib: $(alltypes-$(lib))
ifeq (yes,$(build-shared))
lib-noranlib: $(objpfx)$(lib).so$($(lib).so-version)
endif
else
others: $(alltypes-$(lib))
endif

# The linked shared library is never a dependent of lib-noranlib,
# because linking it will depend on libc.so already being built.
ifneq (,$(filter .os,$(object-suffixes-$(lib))))
others: $(objpfx)$(lib).so$($(lib).so-version)
endif


# Use o-iterator.mk to generate a rule for each flavor of library.
ifneq (,$(filter-out .os .oS,$(object-suffixes-$(lib))))
define o-iterator-doit
$(objpfx)$(patsubst %,$(libtype$o),$(lib:lib%=%)): \
  $(patsubst %,$(objpfx)%$o,\
	     $(filter-out $($(lib)-shared-only-routines),\
			  $(all-$(lib)-routines))); \
	$$(build-extra-lib)
endef
object-suffixes-left = $(filter-out .os .oS,$(object-suffixes-$(lib)))
include $(patsubst %,$(..)o-iterator.mk,$(object-suffixes-left))
endif

ifneq (,$(filter .os,$(object-suffixes-$(lib))))
$(objpfx)$(patsubst %,$(libtype.os),$(lib:lib%=%)): \
  $(patsubst %,$(objpfx)%.os,\
	     $(filter-out $($(lib)-static-only-routines),\
			  $(all-$(lib)-routines)))
	$(build-extra-lib)
endif

ifneq (,$(filter .oS,$(object-suffixes-$(lib))))
$(objpfx)$(patsubst %,$(libtype.oS),$(lib:lib%=%)): \
  $(patsubst %,$(objpfx)%.oS,\
	     $(filter $($(lib)-static-only-routines),\
		      $(all-$(lib)-routines)))
	$(build-extra-lib)
endif

ifeq ($(build-shared),yes)
# Add the version script to the dependencies of the shared library.
$(objpfx)$(lib).so: $(firstword $($(lib)-map) \
				$(addprefix $(common-objpfx), \
					    $(filter $(lib).map, \
						     $(version-maps))))
endif

endif

# This will define `libof-ROUTINE := LIB' for each of the routines.
cpp-srcs-left := $($(lib)-routines) $($(lib)-sysdep_routines)
ifneq (,$(cpp-srcs-left))
include $(patsubst %,$(..)libof-iterator.mk,$(cpp-srcs-left))
endif
