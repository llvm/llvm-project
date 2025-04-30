#!/bin/sh
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

grep -i "$1" $2 > $2.grep
if test $? = 1 ; then
    if [ $3 ]; then
      echo "File $2 does not contain the string that contains the failure"
    else
      echo "File $2 does not contain the string $1."
    fi
    echo " Test PASSES" 
    echo " PASS "
else
    echo "File $2 contains the string $1."
    echo " Test FAILS" 
fi
