#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe83  ########


fe83: fe83.$(OBJX)
	

fe83.$(OBJX):  $(SRC)/fe83.f90
	-$(RM) fe83.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe83.f90 -o fe83.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe83.$(OBJX) check.$(OBJX) $(LIBS) -o fe83.$(EXESUFFIX)


fe83.run: fe83.$(OBJX)
	@echo ------------------------------------ executing test fe83
	fe83.$(EXESUFFIX)
run: fe83.$(OBJX)
	@echo ------------------------------------ executing test fe83
	fe83.$(EXESUFFIX)

verify:	;
build:	fe83.$(OBJX)
