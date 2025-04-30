#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop315b  ########


oop315b: run
	

build:  $(SRC)/oop315b.f90
	-$(RM) oop315b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop315b.f90 -o oop315b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop315b.$(OBJX) check.$(OBJX) $(LIBS) -o oop315b.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop315b
	oop315b.$(EXESUFFIX)

verify: ;

