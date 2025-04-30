#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe101  ########


fe101: fe101.run
	

fe101.$(OBJX):  $(SRC)/fe101.f90
	-$(RM) fe101.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe101.f90 -o fe101.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe101.$(OBJX) check.$(OBJX) $(LIBS) -o fe101.$(EXESUFFIX)


fe101.run: fe101.$(OBJX)
	@echo ------------------------------------ executing test fe101
	fe101.$(EXESUFFIX)

build:	fe101.$(OBJX)

verify:	;

run:	 fe101.$(OBJX)
	@echo ------------------------------------ executing test fe101
	fe101.$(EXESUFFIX)
