# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
########## Make rule for test oop720  ########


oop720: run
	

build:  $(SRC)/oop720.f90
	-$(RM) oop720.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop720.f90 -o oop720.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop720.$(OBJX) check.$(OBJX) $(LIBS) -o oop720.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop720
	oop720.$(EXESUFFIX)

verify: ;

