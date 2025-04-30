#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop752  ########


oop752: run
	

build:  $(SRC)/oop752.f90
	-$(RM) oop752.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop752.f90 -o oop752.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop752.$(OBJX) check.$(OBJX) $(LIBS) -o oop752.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop752
	oop752.$(EXESUFFIX)

verify: ;

