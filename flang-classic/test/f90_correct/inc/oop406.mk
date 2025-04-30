#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop406  ########


oop406: run
	

build:  $(SRC)/oop406.f90
	-$(RM) oop406.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop406.f90 -o oop406.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop406.$(OBJX) check.$(OBJX) $(LIBS) -o oop406.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop406
	oop406.$(EXESUFFIX)

verify: ;

