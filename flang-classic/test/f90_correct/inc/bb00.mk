#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bb00  ########


bb00: run
	

build:  $(SRC)/bb00.f
	-$(RM) bb00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bb00.f -o bb00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bb00.$(OBJX) check.$(OBJX) $(LIBS) -o bb00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bb00
	bb00.$(EXESUFFIX)

verify: ;

bb00.run: run

