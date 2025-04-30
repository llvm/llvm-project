#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt19  ########


dt19: run
	

build:  $(SRC)/dt19.f90
	-$(RM) dt19.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt19.f90 -o dt19.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt19.$(OBJX) check.$(OBJX) $(LIBS) -o dt19.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt19
	dt19.$(EXESUFFIX)

verify: ;

dt19.run: run

