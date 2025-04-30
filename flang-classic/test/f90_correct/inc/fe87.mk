#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe87  ########


fe87: fe87.$(OBJX)
	

fe87.$(OBJX):  $(SRC)/fe87.f90
	-$(RM) fe87.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe87.f90 -o fe87.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe87.$(OBJX) check.$(OBJX) $(LIBS) -o fe87.$(EXESUFFIX)


fe87.run: fe87.$(OBJX)
	@echo ------------------------------------ executing test fe87
	fe87.$(EXESUFFIX)

run: fe87.$(OBJX)
	@echo ------------------------------------ executing test fe87
	fe87.$(EXESUFFIX)
build:	fe87.$(OBJX)
verify:	;
