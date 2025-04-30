#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt17  ########


dt17: run
	

build:  $(SRC)/dt17.f90
	-$(RM) dt17.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt17.f90 -o dt17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt17.$(OBJX) check.$(OBJX) $(LIBS) -o dt17.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt17
	dt17.$(EXESUFFIX)

verify: ;

dt17.run: run

