# Configurations to build
#
# This section must define:
#   Description - A description of this target.
#   Configs - The names of each configuration to build; this is used to build
#             multiple libraries inside a single configuration file (for
#             example, Debug and Release builds, or builds with and without
#             software floating point).
#
# This section must define one of:
#   UniveralArchs - A list of architectures to build for, when using universal build
#           support (e.g., on Darwin). This should only be used to build fat
#           libraries, simply building multiple libraries for different
#           architectures should do so using distinct configs, with the
#           appropriate choices for CC and CFLAGS.
#
#   Arch - The target architecture; this must match the compiler-rt name for the
#          architecture and is used to find the appropriate function
#          implementations.
#
# When not universal builds, this section may define:
#   Arch.<Config Name> - Set the target architecture on a per-config basis.

Description := Target for building universal libraries for Darwin.

Configs := Debug Release Profile
UniversalArchs := i386 ppc x86_64

# Platform Options
#
# This section may override any of the variables in make/options.mk, using:
#   <Option Name> := ... option value ...
#
# See make/options.mk for the available options and their meanings. Options can
# be override on a per-config, per-arch, or per-config-and-arch basis using:
#   <Option Name>.<Config Name> := ...
#   <Option Name>.<Arch Name> := ...
#   <Option Name>.<Config Name>.<Arch Name> := ...

CC := gcc

CFLAGS := -Wall -Werror
CFLAGS.Debug := $(CFLAGS) -g
CFLAGS.Release := $(CFLAGS) -O3 -fomit-frame-pointer
CFLAGS.Profile := $(CFLAGS) -pg -g

FUNCTIONS.i386 := $(CommonFunctions) $(ArchFunctions.i386)
FUNCTIONS.ppc := $(CommonFunctions) $(ArchFunctions.ppc)
FUNCTIONS.x86_64 := $(CommonFunctions) $(ArchFunctions.x86_64)
FUNCTIONS.armv5 := $(CommonFunctions) $(ArchFunctions.armv5)
FUNCTIONS.armv6 := $(CommonFunctions) $(ArchFunctions.armv6)
FUNCTIONS.armv7 := $(CommonFunctions) $(ArchFunctions.armv7)

OPTIMIZED.Debug := 0

VISIBILITY_HIDDEN := 1
