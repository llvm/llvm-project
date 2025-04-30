#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop315a  ########


oop315a: run
	

build:  $(SRC)/oop315a.f90
	-$(RM) oop315a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop315a.f90 -o oop315a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop315a.$(OBJX) check.$(OBJX) $(LIBS) -o oop315a.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop315a
	oop315a.$(EXESUFFIX)

verify: ;

