#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop438  ########


oop438: run
	

build:  $(SRC)/oop438.f90
	-$(RM) oop438.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop438.f90 -o oop438.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop438.$(OBJX) check.$(OBJX) $(LIBS) -o oop438.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop438
	oop438.$(EXESUFFIX)

verify: ;

