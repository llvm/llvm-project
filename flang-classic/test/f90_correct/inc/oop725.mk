#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
########## Make rule for test oop725  ########


oop725: run
	

build:  $(SRC)/oop725.f90
	-$(RM) oop725.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop725.f90 -o oop725.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop725.$(OBJX) check.$(OBJX) $(LIBS) -o oop725.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop725
	oop725.$(EXESUFFIX)

verify: ;

