#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe82  ########


fe82: run
	

build:  $(SRC)/fe82.f90
	-$(RM) fe82.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe82.f90 -o fe82.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe82.$(OBJX) check.$(OBJX) $(LIBS) -o fe82.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe82
	fe82.$(EXESUFFIX)

verify: ;

fe82.run: run

