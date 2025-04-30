#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop773  ########


oop773: run
	

build:  $(SRC)/oop773.f90
	-$(RM) oop773.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop773.f90 -o oop773.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop773.$(OBJX) check.$(OBJX) $(LIBS) -o oop773.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop773
	oop773.$(EXESUFFIX)

verify: ;

