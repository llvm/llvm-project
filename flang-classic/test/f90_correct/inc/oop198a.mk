#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop198a  ########


oop198a: run
	

build:  $(SRC)/oop198a.f90
	-$(RM) oop198a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop198a.f90 -o oop198a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop198a.$(OBJX) check.$(OBJX) $(LIBS) -o oop198a.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop198a
	oop198a.$(EXESUFFIX)

verify: ;

