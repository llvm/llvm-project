###
# Configuration variables.

OS := $(shell uname)

# Assume make is always run from top-level of source directory. Note than an
# Apple style build overrides these variables later in the makefile.
ProjSrcRoot := $(shell pwd)
ProjObjRoot := $(ProjSrcRoot)

###
# Tool configuration variables.

# FIXME: LLVM uses autoconf/mkinstalldirs ?
MKDIR := mkdir -p
DATE := date
LIPO := lipo
CP := cp

VERBOSE := 0
DEBUGMAKE :=

###
# Automatic and derived variables.

# Adjust settings for verbose mode
ifneq ($(VERBOSE),1)
  Verb := @
else
  Verb :=
endif

Echo := @echo
ifndef Summary
  Summary = $(Echo)
endif
