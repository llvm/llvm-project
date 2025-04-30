#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop420  ########


oop420: run
	

build:  $(SRC)/oop420.f90
	-$(RM) oop420.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop420.f90 -o oop420.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop420.$(OBJX) check.$(OBJX) $(LIBS) -o oop420.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop420
	oop420.$(EXESUFFIX)

verify: ;

