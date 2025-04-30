#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop392  ########


oop392: run
	

build:  $(SRC)/oop392.f90
	-$(RM) oop392.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop392.f90 -o oop392.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop392.$(OBJX) check.$(OBJX) $(LIBS) -o oop392.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop392
	oop392.$(EXESUFFIX)

verify: ;

