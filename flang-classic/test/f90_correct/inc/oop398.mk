#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop398  ########


oop398: run
	

build:  $(SRC)/oop398.f90
	-$(RM) oop398.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop398.f90 -o oop398.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop398.$(OBJX) check.$(OBJX) $(LIBS) -o oop398.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop398
	oop398.$(EXESUFFIX)

verify: ;

