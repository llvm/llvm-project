#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop354  ########


oop354: run
	

build:  $(SRC)/oop354.f90
	-$(RM) oop354.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop354.f90 -o oop354.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop354.$(OBJX) check.$(OBJX) $(LIBS) -o oop354.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop354
	oop354.$(EXESUFFIX)

verify: ;

