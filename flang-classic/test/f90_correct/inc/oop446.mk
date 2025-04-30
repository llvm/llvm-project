#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop446  ########


oop446: run
	

build:  $(SRC)/oop446.f90
	-$(RM) oop446.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop446.f90 -o oop446.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop446.$(OBJX) check.$(OBJX) $(LIBS) -o oop446.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop446
	oop446.$(EXESUFFIX)

verify: ;

