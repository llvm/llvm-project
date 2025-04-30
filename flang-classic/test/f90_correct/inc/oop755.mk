#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop755  ########


oop755: run
	

build:  $(SRC)/oop755.f90
	-$(RM) oop755.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop755.f90 -o oop755.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop755.$(OBJX) check.$(OBJX) $(LIBS) -o oop755.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop755
	oop755.$(EXESUFFIX)

verify: ;

