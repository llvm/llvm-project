# This file is intended to be included from each subdirectory makefile.
#
# Subdirectory makefiles must define:
#   SubDirs - The subdirectories to traverse.
#   ObjNames - The objects available in that directory.
#   Implementation - The library configuration the objects should go in (Generic or
#                    Optimized)
#   Dependencies - Any dependences for the object files.
#
# Subdirectory makefiles may define:
#   OnlyArchs - Only build the objects for the listed archs.
#   OnlyConfigs - Only build the objects for the listed configurations.

ifeq ($(Dir),)
  $(error "No Dir variable defined.")
endif

###
# Include child makefile fragments

# The list of variables which are intended to be overridden in a subdirectory
# makefile.
RequiredSubdirVariables := SubDirs ObjNames Implementation Dependencies
OptionalSubdirVariables := OnlyArchs OnlyConfigs

# Template: subdir_traverse_template subdir
define subdir_traverse_template
$(call Set,Dir,$(1))
ifneq ($(DEBUGMAKE),)
  $$(info MAKE: $(Dir): Processing subdirectory)
endif

# Construct the variable key for this directory.
$(call Set,DirKey,SubDir.$(subst .,,$(subst /,__,$(1))))
$(call Append,SubDirKeys,$(DirKey))
$(call Set,$(DirKey).Dir,$(Dir))

# Reset subdirectory specific variables to sentinel value.
$$(foreach var,$$(RequiredSubdirVariables) $$(OptionalSubdirVariables),\
  $$(call Set,$$(var),UNDEFINED))

# Get the subdirectory variables.
include $(1)/Makefile.mk

ifeq ($(DEBUGMAKE),2)
$$(foreach var,$(RequiredSubdirVariables) $(OptionalSubdirVariables),\
  $$(if $$(call strneq,UNDEFINED,$$($$(var))), \
	$$(info MAKE: $(Dir): $$(var) is defined), \
	$$(info MAKE: $(Dir): $$(var) is undefined)))
endif

# Check for undefined required variables, and unset sentinel value from optional
# variables.
$$(foreach var,$(RequiredSubdirVariables),\
  $$(if $$(call strneq,UNDEFINED,$$($$(var))),, \
	$$(error $(Dir): variable '$$(var)' was not undefined)))
$$(foreach var,$(OptionalSubdirVariables),\
  $$(if $$(call strneq,UNDEFINED,$$($$(var))),, \
	$$(call Set,$$(var),)))

# Collect all subdirectory variables for subsequent use.
$$(foreach var,$(RequiredSubdirVariables) $(OptionalSubdirVariables),\
  $$(call Set,$(DirKey).$$(var),$$($$(var))))

# Recurse.
include make/subdir.mk

# Restore directory variable, for cleanliness.
$$(call Set,Dir,$(1))

ifneq ($(DEBUGMAKE),)
  $$(info MAKE: $$(Dir): Done processing subdirectory)
endif
endef

# Evaluate this now so we do not have to worry about order of evaluation.

SubDirsList := $(strip \
  $(if $(call streq,.,$(Dir)),\
       $(SubDirs),\
       $(SubDirs:%=$(Dir)/%)))
ifeq ($(SubDirsList),)
else
  ifneq ($(DEBUGMAKE),)
    $(info MAKE: Descending into subdirs: $(SubDirsList))
  endif

  $(foreach subdir,$(SubDirsList),\
	$(eval $(call subdir_traverse_template,$(subdir))))
endif
