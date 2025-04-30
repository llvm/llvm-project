#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cat $1 | awk -v FS="[(,]" ' \
    BEGIN{error=0} \
      /^MTHINTRIN/{ \
        n = $2" "$3" "$4; \
        if (n in defs) { \
	  ++error; print NR " Duplicate:" $0; \
        } else { \
	  defs[n] = $0 \
	} \
      } \
    END{exit(error?1:0)}'
