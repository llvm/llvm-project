#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop473  ########


oop473: run
	

build:  $(SRC)/oop473.f90
	-$(RM) oop473.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop473.f90 -o oop473.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop473.$(OBJX) check.$(OBJX) $(LIBS) -o oop473.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop473
	oop473.$(EXESUFFIX)

verify: ;

