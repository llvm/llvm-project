#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop471  ########


oop471: run
	

build:  $(SRC)/oop471.f90
	-$(RM) oop471.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop471.f90 -o oop471.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop471.$(OBJX) check.$(OBJX) $(LIBS) -o oop471.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop471
	oop471.$(EXESUFFIX)

verify: ;

