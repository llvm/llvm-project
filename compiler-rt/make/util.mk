# Generic Makefile Utilities

###
# Utility functions

# Function: streq LHS RHS
#
# Return "true" if LHS == RHS, otherwise "".
#
# LHS == RHS <=> (LHS subst RHS is empty) and (RHS subst LHS is empty)
streq = $(if $(1),$(if $(subst $(1),,$(2))$(subst $(2),,$(1)),,true),$(if $(2),,true))

# Function: strneq LHS RHS
#
# Return "true" if LHS != RHS, otherwise "".
strneq = $(if $(call streq,$(1),$(2)),,true)

# Function: contains list item
#
# Return "true" if 'list' contains the value 'item'.
contains = $(if $(strip $(foreach i,$(1),$(if $(call streq,$(2),$(i)),T,))),true,)

# Function: is_subset a b
# Return "true" if 'a' is a subset of 'b'.
is_subset = $(if $(strip $(set_difference $(1),$(2))),,true)

# Function: set_difference a b
# Return a - b.
set_difference = $(foreach i,$(1),$(if $(call contains,$(2),$(i)),,$(i)))

# Function: Set variable value
#
# Set the given make variable to the given value.
Set = $(eval $(1) := $(2))

# Function: Append variable value
#
# Append the given value to the given make variable.
Append = $(eval $(1) += $(2))

# Function: IsDefined variable
#
# Check whether the given variable is defined.
IsDefined = $(call strneq,undefined,$(flavor $(1)))

# Function: IsUndefined variable
#
# Check whether the given variable is undefined.
IsUndefined = $(call streq,undefined,$(flavor $(1)))

# Function: VarOrDefault variable default-value
#
# Get the value of the given make variable, or the default-value if the variable
# is undefined.
VarOrDefault = $(if $(call IsDefined,$(1)),$($(1)),$(2))

# Function: CheckValue variable
#
# Print the name, definition, and value of a variable, for testing make
# utilities.
#
# Example:
#   foo = $(call streq,a,a)
#   $(call CheckValue,foo)
# Example Output:
#   CHECKVALUE: foo: $(call streq,,) - true
CheckValue = $(info CHECKVALUE: $(1): $(value $(1)) - $($(1)))

# Function: CopyVariable src dst
#
# Copy the value of the variable 'src' to 'dst', taking care to not define 'dst'
# if 'src' is undefined. The destination variable must be undefined.
CopyVariable = \
  $(call AssertValue,$(call IsUndefined,$(2)),destination is already defined)\
  $(if $(call IsUndefined,$(1)),,\
       $(call Set,$(2),$($(1))))

# Function: Assert value message
#
# Check that a value is true, or give an error including the given message
Assert = $(if $(1),,\
           $(error Assertion failed: $(2)))

# Function: AssertEqual variable expected-value
#
# Check that the value of a variable is 'expected-value'.
AssertEqual = \
  $(if $(call streq,$($(1)),$(2)),,\
       $(error Assertion failed: $(1): $(value $(1)) - $($(1)) != $(2)))

# Function: CheckCommandLineOverrides list
#
# Check that all command line variables are in the given list. This routine is
# useful for validating that users aren't trying to override something which
# will not work.
CheckCommandLineOverrides = \
  $(foreach arg,$(MAKEOVERRIDES),\
    $(call Set,varname,$(firstword $(subst =, ,$(arg)))) \
    $(if $(call contains,$(1),$(varname)),,\
      $(error "Invalid command line override: $(1) $(varname) (not supported)")))

###
# Clean up make behavior

# Cancel all suffix rules. We don't want no stinking suffix rules.
.SUFFIXES:

###
# Debugging

# General debugging rule, use 'make print-XXX' to print the definition, value
# and origin of XXX.
make-print-%:
	$(error PRINT: $(value $*) = "$($*)" (from $(origin $*)))
