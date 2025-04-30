#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop429  ########


oop429: run
	

build:  $(SRC)/oop429.f90
	-$(RM) oop429.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop429.f90 -o oop429.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop429.$(OBJX) check.$(OBJX) $(LIBS) -o oop429.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop429
	oop429.$(EXESUFFIX)

verify: ;

