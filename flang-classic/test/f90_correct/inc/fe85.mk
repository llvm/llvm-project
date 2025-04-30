#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe85  ########


fe85: fe85.$(OBJX)
	

fe85.$(OBJX):  $(SRC)/fe85.f90
	-$(RM) fe85.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe85.f90 -o fe85.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe85.$(OBJX) check.$(OBJX) $(LIBS) -o fe85.$(EXESUFFIX)


fe85.run: fe85.$(OBJX)
	@echo ------------------------------------ executing test fe85
	fe85.$(EXESUFFIX)
run: fe85.$(OBJX)
	@echo ------------------------------------ executing test fe85
	fe85.$(EXESUFFIX)

run:	fe85.run
verify:	;
build:	fe85.$(OBJX)
