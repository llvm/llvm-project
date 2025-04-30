#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe81  ########


fe81: run
	

build:  $(SRC)/fe81.f90
	-$(RM) fe81.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe81.f90 -o fe81.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe81.$(OBJX) check.$(OBJX) $(LIBS) -o fe81.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe81
	fe81.$(EXESUFFIX)

verify: ;

fe81.run: run

