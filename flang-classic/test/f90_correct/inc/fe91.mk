#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe91  ########


fe91: run
	

build:  $(SRC)/fe91.f90
	-$(RM) fe91.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe91.f90 -o fe91.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe91.$(OBJX) check.$(OBJX) $(LIBS) -o fe91.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe91
	fe91.$(EXESUFFIX)

verify: ;

fe91.run: run

