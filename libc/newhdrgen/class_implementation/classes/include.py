#!/usr/bin/env python
#
# ====-- Include class for libc function headers --------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class Include:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'#include "{self.name}"'
