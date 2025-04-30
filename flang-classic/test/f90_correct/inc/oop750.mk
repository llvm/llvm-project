#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop750  ########


oop750: run
	

build:  $(SRC)/oop750.f90
	-$(RM) oop750.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop750.f90 -o oop750.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop750.$(OBJX) check.$(OBJX) $(LIBS) -o oop750.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop750
	oop750.$(EXESUFFIX)

verify: ;

