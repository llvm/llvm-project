#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop756  ########


oop756: run
	

build:  $(SRC)/oop756.f90
	-$(RM) oop756.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop756.f90 -o oop756.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop756.$(OBJX) check.$(OBJX) $(LIBS) -o oop756.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop756
	oop756.$(EXESUFFIX)

verify: ;

