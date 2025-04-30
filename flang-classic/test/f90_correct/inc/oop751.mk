#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop751  ########


oop751: run
	

build:  $(SRC)/oop751.f90
	-$(RM) oop751.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop751.f90 -o oop751.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop751.$(OBJX) check.$(OBJX) $(LIBS) -o oop751.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop751
	oop751.$(EXESUFFIX)

verify: ;

