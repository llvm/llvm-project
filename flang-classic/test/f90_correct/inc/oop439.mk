#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop439  ########


oop439: run
	

build:  $(SRC)/oop439.f90
	-$(RM) oop439.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop439.f90 -o oop439.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop439.$(OBJX) check.$(OBJX) $(LIBS) -o oop439.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop439
	oop439.$(EXESUFFIX)

verify: ;

