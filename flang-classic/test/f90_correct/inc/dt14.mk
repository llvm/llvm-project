#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt14  ########


dt14: run
	

build:  $(SRC)/dt14.f90
	-$(RM) dt14.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt14.f90 -o dt14.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt14.$(OBJX) check.$(OBJX) $(LIBS) -o dt14.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt14
	dt14.$(EXESUFFIX)

verify: ;

dt14.run: run

