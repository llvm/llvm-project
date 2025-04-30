#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop775  ########


oop775: run
	

build:  $(SRC)/oop775.f90
	-$(RM) oop775.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop775.f90 -o oop775.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop775.$(OBJX) check.$(OBJX) $(LIBS) -o oop775.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop775
	oop775.$(EXESUFFIX)

verify: ;

