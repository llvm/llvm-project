#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop442  ########


oop442: run
	

build:  $(SRC)/oop442.f90
	-$(RM) oop442.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop442.f90 -o oop442.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop442.$(OBJX) check.$(OBJX) $(LIBS) -o oop442.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop442
	oop442.$(EXESUFFIX)

verify: ;

