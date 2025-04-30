#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cat $1 | awk -v FS=, 'function f(n){print "MTHTMPDEF("n")"}/^MTH/{f($4);f($5);f($6);f($7)}' | sed 's/ //g' | sort | uniq | grep -v -e __math_dispatch_error > $2
