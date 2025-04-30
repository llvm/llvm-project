#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop440  ########


oop440: run
	

build:  $(SRC)/oop440.f90
	-$(RM) oop440.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop440.f90 -o oop440.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop440.$(OBJX) check.$(OBJX) $(LIBS) -o oop440.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop440
	oop440.$(EXESUFFIX)

verify: ;

