#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop731a  ########


oop731a: run
	

build:  $(SRC)/oop731a.f90
	-$(RM) oop731a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop731a.f90 -o oop731a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop731a.$(OBJX) check.$(OBJX) $(LIBS) -o oop731a.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop731a
	oop731a.$(EXESUFFIX)

verify: ;

