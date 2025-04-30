#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop449  ########


oop449: run
	

build:  $(SRC)/oop449.f90
	-$(RM) oop449.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop449.f90 -o oop449.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop449.$(OBJX) check.$(OBJX) $(LIBS) -o oop449.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop449
	oop449.$(EXESUFFIX)

verify: ;

