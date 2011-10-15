# compiler-rt Configuration Support
#
# This should be included following 'lib_util.mk'.

# The simple variables configurations can define.
PlainConfigVariables := Configs Description
PerConfigVariables := UniversalArchs Arch $(AvailableOptions)
RequiredConfigVariables := Configs Description

###
# Load Platforms

# Template: subdir_traverse_template subdir
define load_platform_template
$(call Set,PlatformName,$(basename $(notdir $(1))))
ifneq ($(DEBUGMAKE),)
  $$(info MAKE: $(PlatformName): Loading platform)
endif

# Construct the variable key for this directory.
$(call Set,PlatformKey,Platform.$(PlatformName))
$(call Append,PlatformKeys,$(PlatformKey))
$(call Set,$(PlatformKey).Name,$(PlatformName))
$(call Set,$(PlatformKey).Path,$(1))

# Reset platform specific variables to sentinel value.
$$(foreach var,$(PlainConfigVariables) $(PerConfigVariables),\
  $$(call Set,$$(var),UNDEFINED))
$$(foreach var,$(PerConfigVariables),\
  $$(foreach config,$$(Configs),\
    $$(call Set,$$(var).$$(config),UNDEFINED)))
$$(foreach var,$(PerConfigVariables),\
  $$(foreach arch,$(AvailableArchs),\
    $$(call Set,$$(var).$$(arch),UNDEFINED)))

# Get the platform variables.
include make/options.mk
include $(1)

# Check for undefined required variables.
$$(foreach var,$(RequiredConfigVariables),\
  $$(if $$(call strneq,UNDEFINED,$$($$(var))),, \
	$$(error $(Dir): variable '$$(var)' was not undefined)))

# Check that exactly one of UniversalArchs or Arch was defined.
$$(if $$(and $$(call strneq,UNDEFINED,$$(UniversalArchs)),\
             $$(call strneq,UNDEFINED,$$(Arch))),\
    $$(error '$(1)': cannot define both 'UniversalArchs' and 'Arch'))
$$(if $$(or $$(call strneq,UNDEFINED,$$(UniversalArchs)),\
            $$(call strneq,UNDEFINED,$$(Arch))),,\
    $$(error '$(1)': must define one of 'UniversalArchs' and 'Arch'))

# Collect all the platform variables for subsequent use.
$$(foreach var,$(PlainConfigVariables) $(PerConfigVariables),\
  $$(if $$(call strneq,UNDEFINED,$$($$(var))),\
    $$(call CopyVariable,$$(var),$(PlatformKey).$$(var))))
$$(foreach var,$(PerConfigVariables),\
  $$(foreach config,$$(Configs),\
    $$(if $$(call strneq,UNDEFINED,$$($$(var).$$(config))),\
      $$(call CopyVariable,$$(var).$$(config),$(PlatformKey).$$(var).$$(config))))\
  $$(foreach arch,$(AvailableArchs),\
    $$(if $$(call strneq,UNDEFINED,$$($$(var).$$(arch))),\
      $$(call CopyVariable,$$(var).$$(arch),$(PlatformKey).$$(var).$$(arch))))\
  $$(foreach config,$$(Configs),\
    $$(foreach arch,$(AvailableArchs),\
      $$(if $$(call strneq,UNDEFINED,$$($$(var).$$(config).$$(arch))),\
        $$(call CopyVariable,$$(var).$$(config).$$(arch),\
                $(PlatformKey).$$(var).$$(config).$$(arch))))))

ifneq ($(DEBUGMAKE),)
  $$(info MAKE: $(PlatformName): Done loading platform)
endif
endef

# Evaluate this now so we do not have to worry about order of evaluation.
PlatformFiles := $(wildcard make/platform/*.mk)
ifneq ($(DEBUGMAKE),)
 $(info MAKE: Loading platforms: $(PlatformFiles))
endif

$(foreach file,$(PlatformFiles),\
  $(eval $(call load_platform_template,$(file))))
