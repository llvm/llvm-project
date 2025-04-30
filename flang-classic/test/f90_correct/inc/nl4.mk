#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nl4  ########


nl4: run
	

build:  $(SRC)/nl4.f90
	-$(RM) nl4.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nl4.f90 -o nl4.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nl4.$(OBJX) check.$(OBJX) $(LIBS) -o nl4.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nl4
	nl4.$(EXESUFFIX)

verify: ;

