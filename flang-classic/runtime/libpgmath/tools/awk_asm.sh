#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

awk '/[ \t]+xxpermdi/{l=$0;gsub(",", " ",$0);if ($2 == $3 && $2 == $4 && $5==2) {hdr="#"};$0=l;}{print hdr $0;hdr=""}' $1 > $2
