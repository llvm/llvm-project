#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop205a  ########


oop205a: run
	

build:  $(SRC)/oop205a.f90
	-$(RM) oop205a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop205a.f90 -o oop205a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop205a.$(OBJX) check.$(OBJX) $(LIBS) -o oop205a.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop205a
	oop205a.$(EXESUFFIX)

verify: ;

