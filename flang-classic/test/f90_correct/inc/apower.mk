#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test power opereator  ########


apower: run


build:  $(SRC)/apower.f90
	-$(RM) apower.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/apower.f90 -o apower.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) apower.$(OBJX) check.$(OBJX) $(LIBS) -o apower.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test apower
	apower.$(EXESUFFIX)

verify: ;

apower.run: run

