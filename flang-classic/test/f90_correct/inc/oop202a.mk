#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop202a  ########


oop202a: run
	

build:  $(SRC)/oop202a.f90
	-$(RM) oop202a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop202a.f90 -o oop202a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop202a.$(OBJX) check.$(OBJX) $(LIBS) -o oop202a.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop202a
	oop202a.$(EXESUFFIX)

verify: ;

