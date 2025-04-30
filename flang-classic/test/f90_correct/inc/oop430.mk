#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop430  ########


oop430: run
	

build:  $(SRC)/oop430.f90
	-$(RM) oop430.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop430.f90 -o oop430.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop430.$(OBJX) check.$(OBJX) $(LIBS) -o oop430.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop430
	oop430.$(EXESUFFIX)

verify: ;

