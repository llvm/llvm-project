# Library Utility Functions
#
# This should be included following 'lib_info.mk'.

# Function: GetCNAVar variable-name platform-key config arch
#
# Get a per-config-and-arch variable value.
GetCNAVar = $(strip \
  $(or $($(2).$(1).$(3).$(4)), \
       $($(2).$(1).$(3)), \
       $($(2).$(1).$(4)), \
       $($(2).$(1))))

# Function: SelectFunctionDir config arch function-name optimized
#
# Choose the appropriate implementation directory to use for 'function-name' in
# the configuration 'config' and on given arch.
SelectFunctionDir = $(strip \
  $(call Set,Tmp.SelectFunctionDir,$(call SelectFunctionDirs,$(1),$(2),$(3),$(4)))\
  $(if $(call streq,1,$(words $(Tmp.SelectFunctionDir))),\
       $(Tmp.SelectFunctionDir),\
       $(error SelectFunctionDir: invalid function name "$(3)" ($(strip\
               $(if $(call streq,0,$(words $(Tmp.SelectFunctionDir))),\
                    no such function,\
                    function implemented in multiple directories!!!))))))

# Helper functions that select the entire list of subdirs where a function is
# defined with a certain specificity.
SelectFunctionDirs_Opt_ConfigAndArch = $(strip \
  $(foreach key,$(AvailableIn.$(3)),\
    $(if $(and $(call streq,Optimized,$($(key).Implementation)),\
               $(call contains,$($(key).OnlyConfigs),$(1)),\
               $(call contains,$($(key).OnlyArchs),$(2))),$(key),)))
SelectFunctionDirs_Opt_Config = $(strip \
  $(foreach key,$(AvailableIn.$(3)),\
    $(if $(and $(call streq,Optimized,$($(key).Implementation)),\
               $(call contains,$($(key).OnlyConfigs),$(1))),$(key),)))
SelectFunctionDirs_Opt_Arch = $(strip \
  $(foreach key,$(AvailableIn.$(3)),\
    $(if $(and $(call streq,Optimized,$($(key).Implementation)),\
               $(call contains,$($(key).OnlyArchs),$(2))),$(key),)))
SelectFunctionDirs_Gen = $(strip \
  $(foreach key,$(AvailableIn.$(3)),\
    $(if $(call streq,Generic,$($(key).Implementation)),$(key))))

# Helper function to select the right set of dirs in generic priority order.
SelectFunctions_Gen = \
  $(or $(call SelectFunctionDirs_Gen,$(1),$(2),$(3)),\
       $(call SelectFunctionDirs_Opt_ConfigAndArch,$(1),$(2),$(3)), \
       $(call SelectFunctionDirs_Opt_Config,$(1),$(2),$(3)), \
       $(call SelectFunctionDirs_Opt_Arch,$(1),$(2),$(3)))

# Helper function to select the right set of dirs in optimized priority order.
SelectFunctions_Opt = \
  $(or $(call SelectFunctionDirs_Opt_ConfigAndArch,$(1),$(2),$(3)), \
       $(call SelectFunctionDirs_Opt_Config,$(1),$(2),$(3)), \
       $(call SelectFunctionDirs_Opt_Arch,$(1),$(2),$(3)), \
       $(call SelectFunctionDirs_Gen,$(1),$(2),$(3)))

# Helper function to select the right set of dirs (which should be exactly one)
# for a function.
SelectFunctionDirs = \
  $(if $(call streq,1,$(4)),\
       $(call SelectFunctions_Opt,$(1),$(2),$(3)),\
       $(call SelectFunctions_Gen,$(1),$(2),$(3)))
