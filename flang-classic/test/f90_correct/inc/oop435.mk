#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop435  ########


oop435: run
	

build:  $(SRC)/oop435.f90
	-$(RM) oop435.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop435.f90 -o oop435.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop435.$(OBJX) check.$(OBJX) $(LIBS) -o oop435.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop435
	oop435.$(EXESUFFIX)

verify: ;

