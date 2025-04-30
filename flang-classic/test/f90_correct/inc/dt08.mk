#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt08  ########


dt08: run
	

build:  $(SRC)/dt08.f90
	-$(RM) dt08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt08.f90 -o dt08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt08.$(OBJX) check.$(OBJX) $(LIBS) -o dt08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt08
	dt08.$(EXESUFFIX)

verify: ;

dt08.run: run

