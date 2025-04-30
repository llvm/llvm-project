#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop468  ########


oop468: run
	

build:  $(SRC)/oop468.f90
	-$(RM) oop468.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop468.f90 -o oop468.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop468.$(OBJX) check.$(OBJX) $(LIBS) -o oop468.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop468
	oop468.$(EXESUFFIX)

verify: ;

