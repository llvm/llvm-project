#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nl1  ########


nl1: run
	

build:  $(SRC)/nl1.f90
	-$(RM) nl1.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nl1.f90 -o nl1.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nl1.$(OBJX) check.$(OBJX) $(LIBS) -o nl1.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nl1
	nl1.$(EXESUFFIX)

verify: ;

