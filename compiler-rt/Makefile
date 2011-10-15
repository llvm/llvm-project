SubDirs := lib

# Set default rule before anything else.
all: help

include make/config.mk
include make/util.mk
# If SRCROOT is defined, assume we are doing an Apple style build. We should be
# able to use RC_XBS for this but that is unused during "make installsrc".
ifdef SRCROOT
  include make/AppleBI.mk
endif

# Make sure we don't build with a missing ProjObjRoot.
ifeq ($(ProjObjRoot),)
$(error Refusing to build with empty ProjObjRoot variable)
endif

##############

###
# Rules

###
# Top level targets

# FIXME: Document the available subtargets.
help:
	@echo "usage: make [{VARIABLE=VALUE}*] target"
	@echo
	@echo "User variables:"
	@echo "  VERBOSE=1: Use to show all commands [default=0]"
	@echo
	@echo "Available targets:"
	@echo "  <platform name>: build the libraries for 'platform'"
	@echo "  clean:           clean all configurations"
	@echo "  test:            run unit tests"
	@echo
	@echo "  info-platforms:  list available platforms"
	@echo "  help-devel:      print additional help for developers"
	@echo

help-devel: help
	@echo "Development targets:"
	@echo "  <platform name>-<config name>:"
	@echo "    build the libraries for a single platform config"
	@echo "  <platform name>-<config name>-<arch name>:"
	@echo "    build the libraries for a single config and arch"
	@echo "  info-functions: list available compiler-rt functions"
	@echo "  help-hidden: print help for Makefile debugging"
	@echo

help-hidden: help-devel
	@echo "Debugging variables:"
	@echo "  DEBUGMAKE=1: enable some Makefile logging [default=]"
	@echo "           =2: enable more Makefile logging"
	@echo
	@echo "Debugging targets:"
	@echo "  make-print-FOO: print information on the variable 'FOO'"
	@echo

info-functions:
	@echo "compiler-rt Available Functions"
	@echo
	@echo "All Functions: $(AvailableFunctions)"
	@$(foreach fn,$(AvailableFunctions),\
	  printf "  %-20s - available in (%s)\n" $(fn)\
	    "$(foreach key,$(AvailableIn.$(fn)),$($(key).Dir))";)

info-platforms:
	@echo "compiler-rt Available Platforms"
	@echo
	@echo "Platforms:"
	@$(foreach key,$(PlatformKeys),\
	  printf "  %s - from '%s'\n" $($(key).Name) $($(key).Path);\
	  printf "    %s\n" "$($(key).Description)";\
	  printf "    Configurations: %s\n\n" "$($(key).Configs)";)

# Provide default clean target which is extended by other templates.
.PHONY: clean
clean::

# Test
.PHONY: test
test:
	cd test/Unit && ./test

###
# Directory handling magic.

# Create directories as needed, and timestamp their creation.
%/.dir:
	$(Summary) "  MKDIR:     $*"
	$(Verb) $(MKDIR) $* > /dev/null
	$(Verb) $(DATE) > $@

# Remove directories
%/.remove:
	$(Verb) $(RM) -r $*

###
# Include child makefile fragments

Dir := .
include make/subdir.mk
include make/lib_info.mk
include make/lib_util.mk
include make/lib_platforms.mk

###
# Define Platform Rules

define PerPlatform_template
$(call Set,Tmp.Key,$(1))
$(call Set,Tmp.Name,$($(Tmp.Key).Name))
$(call Set,Tmp.Configs,$($(Tmp.Key).Configs))
$(call Set,Tmp.ObjPath,$(ProjObjRoot)/$(Tmp.Name))

# Top-Level Platform Target
$(Tmp.Name):: $(Tmp.Configs:%=$(Tmp.ObjPath)/%/libcompiler_rt.a)
.PHONY: $(Tmp.Name)

clean::
	$(Verb) rm -rf $(Tmp.ObjPath)

# Per-Config Libraries
$(foreach config,$(Tmp.Configs),\
  $(call PerPlatformConfig_template,$(config)))
endef

define PerPlatformConfig_template
$(call Set,Tmp.Config,$(1))
$(call Set,Tmp.ObjPath,$(ProjObjRoot)/$(Tmp.Name)/$(Tmp.Config))

# Compute the archs to build, depending on whether this is a universal build or
# not.
$(call Set,Tmp.ArchsToBuild,\
  $(if $(call IsDefined,$(Tmp.Key).UniversalArchs),\
       $(strip \
         $(or $($(Tmp.Key).UniversalArchs.$(Tmp.Config)),\
              $($(Tmp.Key).UniversalArchs))),\
       $(call VarOrDefault,$(Tmp.Key).Arch.$(Tmp.Config),$($(Tmp.Key).Arch))))

# Copy or lipo to create the per-config library.
$(call Set,Tmp.Inputs,$(Tmp.ArchsToBuild:%=$(Tmp.ObjPath)/%/libcompiler_rt.a))
$(Tmp.ObjPath)/libcompiler_rt.a: $(Tmp.Inputs) $(Tmp.ObjPath)/.dir
	$(Summary) "  FINAL-ARCHIVE: $(Tmp.Name)/$(Tmp.Config): $$@"
	-$(Verb) $(RM) $$@
	$(if $(call streq,1,$(words $(Tmp.ArchsToBuild))), \
	  $(Verb) $(CP) $(Tmp.Inputs) $$@, \
	  $(Verb) $(LIPO) -create -output $$@ $(Tmp.Inputs))
.PRECIOUS: $(Tmp.ObjPath)/.dir

# Per-Config Targets
$(Tmp.Name)-$(Tmp.Config):: $(Tmp.ObjPath)/libcompiler_rt.a
.PHONY: $(Tmp.Name)-$(Tmp.Config)

# Per-Config-Arch Libraries
$(foreach arch,$(Tmp.ArchsToBuild),\
  $(call PerPlatformConfigArch_template,$(arch)))
endef

define PerPlatformConfigArch_template
$(call Set,Tmp.Arch,$(1))
$(call Set,Tmp.ObjPath,$(ProjObjRoot)/$(Tmp.Name)/$(Tmp.Config)/$(Tmp.Arch))
$(call Set,Tmp.Functions,$(strip \
  $(call GetCNAVar,FUNCTIONS,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.Optimized,$(strip \
  $(call GetCNAVar,OPTIMIZED,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.AR,$(strip \
  $(call GetCNAVar,AR,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.ARFLAGS,$(strip \
  $(call GetCNAVar,ARFLAGS,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.RANLIB,$(strip \
  $(call GetCNAVar,RANLIB,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.RANLIBFLAGS,$(strip \
  $(call GetCNAVar,RANLIBFLAGS,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))

# Compute the object inputs for this library.
$(call Set,Tmp.Inputs,\
  $(foreach fn,$(sort $(Tmp.Functions)),\
    $(call Set,Tmp.FnDir,\
      $(call SelectFunctionDir,$(Tmp.Config),$(Tmp.Arch),$(fn),$(Tmp.Optimized)))\
    $(Tmp.ObjPath)/$(Tmp.FnDir)/$(fn).o))
$(Tmp.ObjPath)/libcompiler_rt.a: $(Tmp.Inputs) $(Tmp.ObjPath)/.dir
	$(Summary) "  ARCHIVE:   $(Tmp.Name)/$(Tmp.Config)/$(Tmp.Arch): $$@"
	-$(Verb) $(RM) $$@
	$(Verb) $(Tmp.AR) $(Tmp.ARFLAGS) $$@ $(Tmp.Inputs)
	$(Verb) $(Tmp.RANLIB) $(Tmp.RANLIBFLAGS) $$@
.PRECIOUS: $(Tmp.ObjPath)/.dir

# Per-Config-Arch Targets
$(Tmp.Name)-$(Tmp.Config)-$(Tmp.Arch):: $(Tmp.ObjPath)/libcompiler_rt.a
.PHONY: $(Tmp.Name)-$(Tmp.Config)-$(Tmp.Arch)

# Per-Config-Arch-SubDir Objects
$(foreach key,$(SubDirKeys),\
  $(call PerPlatformConfigArchSubDir_template,$(key)))
endef

define PerPlatformConfigArchSubDir_template
$(call Set,Tmp.SubDirKey,$(1))
$(call Set,Tmp.SubDir,$($(Tmp.SubDirKey).Dir))
$(call Set,Tmp.SrcPath,$(ProjSrcRoot)/$(Tmp.SubDir))
$(call Set,Tmp.ObjPath,$(ProjObjRoot)/$(Tmp.Name)/$(Tmp.Config)/$(Tmp.Arch)/$(Tmp.SubDirKey))
$(call Set,Tmp.Dependencies,$($(Tmp.SubDirKey).Dependencies))
$(call Set,Tmp.CC,$(strip \
  $(call GetCNAVar,CC,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.KERNEL_USE,$(strip \
  $(call GetCNAVar,KERNEL_USE,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.VISIBILITY_HIDDEN,$(strip \
  $(call GetCNAVar,VISIBILITY_HIDDEN,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))
$(call Set,Tmp.CFLAGS,$(strip \
  $(if $(call IsDefined,$(Tmp.Key).UniversalArchs),-arch $(Tmp.Arch),)\
  $(if $(call streq,$(Tmp.VISIBILITY_HIDDEN),1),\
       -fvisibility=hidden -DVISIBILITY_HIDDEN,)\
  $(if $(call streq,$(Tmp.KERNEL_USE),1),\
       -mkernel -DKERNEL_USE,)\
  $(call GetCNAVar,CFLAGS,$(Tmp.Key),$(Tmp.Config),$(Tmp.Arch))))

$(Tmp.ObjPath)/%.o: $(Tmp.SrcPath)/%.s $(Tmp.Dependencies) $(Tmp.ObjPath)/.dir
	$(Summary) "  ASSEMBLE:  $(Tmp.Name)/$(Tmp.Config)/$(Tmp.Arch): $$<"
	$(Verb) $(Tmp.CC) $(Tmp.CFLAGS)  -c -o $$@ $$<
$(Tmp.ObjPath)/%.o: $(Tmp.SrcPath)/%.S $(Tmp.Dependencies) $(Tmp.ObjPath)/.dir
	$(Summary) "  ASSEMBLE:  $(Tmp.Name)/$(Tmp.Config)/$(Tmp.Arch): $$<"
	$(Verb) $(Tmp.CC) $(Tmp.CFLAGS) -c -o $$@ $$<
$(Tmp.ObjPath)/%.o: $(Tmp.SrcPath)/%.c $(Tmp.Dependencies) $(Tmp.ObjPath)/.dir
	$(Summary) "  COMPILE:   $(Tmp.Name)/$(Tmp.Config)/$(Tmp.Arch): $$<"
	$(Verb) $(Tmp.CC) $(Tmp.CFLAGS) -c -o $$@ $$<
.PRECIOUS: $(Tmp.ObjPath)/.dir

endef

# Run templates.
$(foreach key,$(PlatformKeys),\
  $(eval $(call PerPlatform_template,$(key))))

###

ifneq ($(DEBUGMAKE),)
  $(info MAKE: Done processing Makefile)
  $(info  )
endif
