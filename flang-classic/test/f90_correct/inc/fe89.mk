#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe89  ########


fe89: fe89.$(OBJX)
	

fe89.$(OBJX):  $(SRC)/fe89.f90
	-$(RM) fe89.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe89.f90 -o fe89.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe89.$(OBJX) check.$(OBJX) $(LIBS) -o fe89.$(EXESUFFIX)


fe89.run: fe89.$(OBJX)
	@echo ------------------------------------ executing test fe89
	fe89.$(EXESUFFIX)
run: fe89.$(OBJX)
	@echo ------------------------------------ executing test fe89
	fe89.$(EXESUFFIX)

verify:	;
build:	fe89.$(OBJX)
