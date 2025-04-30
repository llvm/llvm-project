#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nl2  ########


nl2: run
	

build:  $(SRC)/nl2.f90
	-$(RM) nl2.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nl2.f90 -o nl2.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nl2.$(OBJX) check.$(OBJX) $(LIBS) -o nl2.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nl2
	nl2.$(EXESUFFIX)

verify: ;

