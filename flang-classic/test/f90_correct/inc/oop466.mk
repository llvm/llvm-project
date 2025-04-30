#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop466  ########


oop466: run
	

build:  $(SRC)/oop466.f90
	-$(RM) oop466.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop466.f90 -o oop466.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop466.$(OBJX) check.$(OBJX) $(LIBS) -o oop466.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop466
	oop466.$(EXESUFFIX)

verify: ;

