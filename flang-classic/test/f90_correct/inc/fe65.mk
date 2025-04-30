#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe65  ########


fe65: run
	

build:  $(SRC)/fe65.f90
	-$(RM) fe65.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe65.f90 -o fe65.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe65.$(OBJX) check.$(OBJX) $(LIBS) -o fe65.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe65
	fe65.$(EXESUFFIX)

verify: ;

fe65.run: run

