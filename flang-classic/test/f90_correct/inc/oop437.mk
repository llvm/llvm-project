#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop437  ########


oop437: run
	

build:  $(SRC)/oop437.f90
	-$(RM) oop437.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop437.f90 -o oop437.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop437.$(OBJX) check.$(OBJX) $(LIBS) -o oop437.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop437
	oop437.$(EXESUFFIX)

verify: ;

