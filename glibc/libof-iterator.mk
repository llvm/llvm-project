# This file is included several times in a row, once for each element
# $(cpp-src) of $(cpp-srcs-left).  It sets libof-$(cpp-src) to $(lib)
# for each.

cpp-src := $(firstword $(cpp-srcs-left))
cpp-srcs-left := $(filter-out $(cpp-src),$(cpp-srcs-left))

libof-$(notdir $(cpp-src)) := $(lib)
