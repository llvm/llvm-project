#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sind  ########


sind: run


build:  $(SRC)/sind.f08
	-$(RM) sind.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sind.f08 -o sind.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sind.$(OBJX) check.$(OBJX) $(LIBS) -o sind.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sind
	sind.$(EXESUFFIX)

verify: ;
