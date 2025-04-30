#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test if01  ########


if01: run


build:  $(SRC)/if01.f08
	-$(RM) if01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/if01.f08 -o if01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) if01.$(OBJX) check.$(OBJX) $(LIBS) -o if01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test if01
	if01.$(EXESUFFIX)

verify: ;
