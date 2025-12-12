# ====-- Symbol class for libc function headers----------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from functools import total_ordering


@total_ordering
class Symbol:
    """
    Symbol is the common superclass for each kind of entity named by an
    identifier.  It provides the name field, and defines sort ordering,
    hashing, and equality based only on the name.  The sorting is pretty
    presentation order for identifiers, which is to say it first sorts
    lexically but ignores leading underscores and secondarily sorts with the
    fewest underscores first.
    """

    def __init__(self, name):
        assert name
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()

    def name_without_underscores(self):
        return self.name.lstrip("_")

    def name_sort_key(self):
        ident = self.name_without_underscores()
        return ident, len(self.name) - len(ident)

    def __lt__(self, other):
        return self.name_sort_key() < other.name_sort_key()
