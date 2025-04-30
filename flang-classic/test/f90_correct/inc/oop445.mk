#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop445  ########


oop445: run
	

build:  $(SRC)/oop445.f90
	-$(RM) oop445.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop445.f90 -o oop445.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop445.$(OBJX) check.$(OBJX) $(LIBS) -o oop445.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop445
	oop445.$(EXESUFFIX)

verify: ;

