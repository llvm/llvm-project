#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop444  ########


oop444: run
	

build:  $(SRC)/oop444.f90
	-$(RM) oop444.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop444.f90 -o oop444.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop444.$(OBJX) check.$(OBJX) $(LIBS) -o oop444.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop444
	oop444.$(EXESUFFIX)

verify: ;

