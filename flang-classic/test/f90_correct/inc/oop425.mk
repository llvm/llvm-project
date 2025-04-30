#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop425  ########


oop425: run
	

build:  $(SRC)/oop425.f90
	-$(RM) oop425.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop425.f90 -o oop425.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop425.$(OBJX) check.$(OBJX) $(LIBS) -o oop425.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop425
	oop425.$(EXESUFFIX)

verify: ;

