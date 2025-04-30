#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop472  ########


oop472: run
	

build:  $(SRC)/oop472.f90
	-$(RM) oop472.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop472.f90 -o oop472.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop472.$(OBJX) check.$(OBJX) $(LIBS) -o oop472.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop472
	oop472.$(EXESUFFIX)

verify: ;

