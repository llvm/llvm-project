#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
########## Make rule for test oop723  ########


oop723: run
	

build:  $(SRC)/oop723.f90
	-$(RM) oop723.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop723.f90 -o oop723.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop723.$(OBJX) check.$(OBJX) $(LIBS) -o oop723.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop723
	oop723.$(EXESUFFIX)

verify: ;

