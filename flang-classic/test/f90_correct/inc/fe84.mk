#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe84  ########


fe84: fe84.$(OBJX)
	

fe84.$(OBJX):  $(SRC)/fe84.f90
	-$(RM) fe84.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe84.f90 -o fe84.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe84.$(OBJX) check.$(OBJX) $(LIBS) -o fe84.$(EXESUFFIX)


fe84.run: fe84.$(OBJX)
	@echo ------------------------------------ executing test fe84
	fe84.$(EXESUFFIX)
run: fe84.$(OBJX)
	@echo ------------------------------------ executing test fe84
	fe84.$(EXESUFFIX)

verify:	;
build:	fe84.$(OBJX)
