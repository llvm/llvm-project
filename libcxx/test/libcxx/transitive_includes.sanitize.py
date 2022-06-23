#!/usr/bin/env python
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# This script reads lines from standard input and looks for the names of public C++ headers.
# Specifically, it looks for lines of the form 'c++/v1/header' where 'header' is the name
# of a public C++ header, excluding C compatibility headers.

import os
import re
import sys

headers = []
for line in sys.stdin.readlines():
  match = re.search('c\+\+/v[0-9]+/(.+)', line)
  if not match:
    continue

  header = match.group(1)
  if os.path.basename(header).endswith('.h'): # Skip C headers
    continue

  if os.path.basename(header).startswith('__'): # Skip internal headers
    continue

  headers.append(header)

print('\n'.join(sorted(set(headers))))
