#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe90  ########


fe90: run
	

build:  $(SRC)/fe90.f
	-$(RM) fe90.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe90.f -o fe90.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe90.$(OBJX) check.$(OBJX) $(LIBS) -o fe90.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe90
	fe90.$(EXESUFFIX)

verify: ;

fe90.run: run

