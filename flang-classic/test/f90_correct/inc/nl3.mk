#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nl3  ########


nl3: run
	

build:  $(SRC)/nl3.f90
	-$(RM) nl3.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nl3.f90 -o nl3.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nl3.$(OBJX) check.$(OBJX) $(LIBS) -o nl3.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nl3
	nl3.$(EXESUFFIX)

verify: ;

