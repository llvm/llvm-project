#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop754  ########


oop754: run
	

build:  $(SRC)/oop754.f90
	-$(RM) oop754.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop754.f90 -o oop754.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop754.$(OBJX) check.$(OBJX) $(LIBS) -o oop754.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop754
	oop754.$(EXESUFFIX)

verify: ;

