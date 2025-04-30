#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop418  ########


oop418: run
	

build:  $(SRC)/oop418.f90
	-$(RM) oop418.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop418.f90 -o oop418.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop418.$(OBJX) check.$(OBJX) $(LIBS) -o oop418.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop418
	oop418.$(EXESUFFIX)

verify: ;

