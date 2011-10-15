# compiler-rt Library Info
#
# This should be included once the subdirectory information has been loaded, and
# uses the utilities in 'util.mk'.
#
# This defines the following variables describing compiler-rt:
#   AvailableFunctions   - The entire list of function names (unmangled) the
#                          library can provide.
#   CommonFunctions      - The list of generic functions available.
#   ArchFunctions.<arch> - The list of functions commonly available for
#                          'arch'. This does not include any config specific
#                          functions.
#
#   AvailableIn.<function> - The list of subdir keys where 'function' is
#                            defined.

AvailableArchs := $(sort $(foreach key,$(SubDirKeys),\
	$($(key).OnlyArchs)))

AvailableFunctions := $(sort $(foreach key,$(SubDirKeys),\
	$(basename $($(key).ObjNames))))

CommonFunctions := $(sort\
  $(foreach key,$(SubDirKeys),\
    $(if $(call strneq,,$(strip $($(key).OnlyArchs) $($(key).OnlyConfigs))),,\
         $(basename $($(key).ObjNames)))))

# Compute common arch functions.
$(foreach key,$(SubDirKeys),\
  $(if $(call strneq,,$($(key).OnlyConfigs)),,\
    $(foreach arch,$($(key).OnlyArchs),\
      $(call Append,ArchFunctions.$(arch),$(sort \
        $(basename $($(key).ObjNames)))))))

# Compute arch only functions.
$(foreach arch,$(AvailableArchs),\
  $(call Set,ArchFunctions.$(arch),$(sort $(ArchFunctions.$(arch))))\
  $(call Set,ArchOnlyFunctions.$(arch),\
    $(call set_difference,$(ArchFunctions.$(arch)),$(CommonFunctions))))

# Compute lists of where each function is available.
$(foreach key,$(SubDirKeys),\
  $(foreach fn,$(basename $($(key).ObjNames)),\
    $(call Append,AvailableIn.$(fn),$(key))))

# The names of all the available options.
AvailableOptions := AR ARFLAGS \
                    CC CFLAGS FUNCTIONS OPTIMIZED \
                    RANLIB RANLIBFLAGS \
                    VISIBILITY_HIDDEN \
                    KERNEL_USE \
                    STRIP LIPO
