#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe86  ########


fe86: fe86.$(OBJX)
	

fe86.$(OBJX):  $(SRC)/fe86.f90
	-$(RM) fe86.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe86.f90 -o fe86.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe86.$(OBJX) check.$(OBJX) $(LIBS) -o fe86.$(EXESUFFIX)


fe86.run: fe86.$(OBJX)
	@echo ------------------------------------ executing test fe86
	fe86.$(EXESUFFIX)
run: fe86.$(OBJX)
	@echo ------------------------------------ executing test fe86
	fe86.$(EXESUFFIX)

run:	fe86.run
verify:	;
build:	fe86.$(OBJX)
