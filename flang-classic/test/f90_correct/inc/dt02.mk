#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt02  ########


dt02: run
	

build:  $(SRC)/dt02.f90
	-$(RM) dt02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt02.f90 -o dt02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt02.$(OBJX) check.$(OBJX) $(LIBS) -o dt02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt02
	dt02.$(EXESUFFIX)

verify: ;

dt02.run: run

