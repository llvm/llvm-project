#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe88  ########


fe88: fe88.$(OBJX)
	

fe88.$(OBJX):  $(SRC)/fe88.f90
	-$(RM) fe88.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe88.f90 -o fe88.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe88.$(OBJX) check.$(OBJX) $(LIBS) -o fe88.$(EXESUFFIX)


fe88.run: fe88.$(OBJX)
	@echo ------------------------------------ executing test fe88
	fe88.$(EXESUFFIX)
run: fe88.$(OBJX)
	@echo ------------------------------------ executing test fe88
	fe88.$(EXESUFFIX)

verify:	;
build:	fe88.$(OBJX)
