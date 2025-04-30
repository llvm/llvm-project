#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# First arg is path to source, ${CMAKE_CURRENT_SOURCE_DIR} in cmake
# called from the directories that contains the source files
# Second arg is the name of the output file created

if [ "$3" != "True" ]; then
	pattern='/#ifdef TARGET_SUPPORTS_QUADFP/,/#endif/d'
else
	pattern=''
fi

sed \
		"$pattern" \
	$1/mth_128defs.c |
awk \
		'/^MTH_DISPATCH_FUNC/ { \
			f = $1; \
			sub("^MTH_DISPATCH_FUNC\\(", "", f); \
			sub("\\).*", "", f); next; \
		} \
		/^[[:space:]]*_MTH_I_STATS_INC/ { \
			split($0, s, "[(,)]"); \
			print "DO_MTH_DISPATCH_FUNC(" f ", " s[2] \
				", " s[3] ", ", s[4] ")"; f=""; \
		}' \
	> $2
