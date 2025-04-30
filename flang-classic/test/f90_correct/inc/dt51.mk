#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt51  ########


dt51: run
	

build:  $(SRC)/dt51.f90
	-$(RM) dt51.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt51.f90 -o dt51.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt51.$(OBJX) check.$(OBJX) $(LIBS) -o dt51.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt51
	dt51.$(EXESUFFIX)

verify: ;

dt51.run: run

