#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop750a  ########


oop750a: run
	

build:  $(SRC)/oop750a.f90
	-$(RM) oop750a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop750a.f90 -o oop750a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop750a.$(OBJX) check.$(OBJX) $(LIBS) -o oop750a.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop750a
	oop750a.$(EXESUFFIX)

verify: ;

