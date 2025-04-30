#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop198b  ########


oop198b: run
	

build:  $(SRC)/oop198b.f90
	-$(RM) oop198b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop198b.f90 -o oop198b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop198b.$(OBJX) check.$(OBJX) $(LIBS) -o oop198b.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop198b
	oop198b.$(EXESUFFIX)

verify: ;

