#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop407  ########


oop407: run
	

build:  $(SRC)/oop407.f90
	-$(RM) oop407.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop407.f90 -o oop407.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop407.$(OBJX) check.$(OBJX) $(LIBS) -o oop407.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop407
	oop407.$(EXESUFFIX)

verify: ;

