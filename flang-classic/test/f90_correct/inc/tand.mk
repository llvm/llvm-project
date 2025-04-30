#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test tand  ########


tand: run


build:  $(SRC)/tand.f08
	-$(RM) tand.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/tand.f08 -o tand.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) tand.$(OBJX) check.$(OBJX) $(LIBS) -o tand.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test tand
	tand.$(EXESUFFIX)

verify: ;
