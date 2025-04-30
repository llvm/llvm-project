#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop441  ########


oop441: run
	

build:  $(SRC)/oop441.f90
	-$(RM) oop441.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop441.f90 -o oop441.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop441.$(OBJX) check.$(OBJX) $(LIBS) -o oop441.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop441
	oop441.$(EXESUFFIX)

verify: ;

