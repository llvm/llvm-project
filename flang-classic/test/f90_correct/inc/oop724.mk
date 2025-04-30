#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
########## Make rule for test oop724  ########


oop724: run
	

build:  $(SRC)/oop724.f90
	-$(RM) oop724.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop724.f90 -o oop724.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop724.$(OBJX) check.$(OBJX) $(LIBS) -o oop724.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop724
	oop724.$(EXESUFFIX)

verify: ;

