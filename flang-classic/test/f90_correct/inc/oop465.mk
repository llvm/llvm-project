#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop465  ########


oop465: run
	

build:  $(SRC)/oop465.f90
	-$(RM) oop465.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop465.f90 -o oop465.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop465.$(OBJX) check.$(OBJX) $(LIBS) -o oop465.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop465
	oop465.$(EXESUFFIX)

verify: ;

