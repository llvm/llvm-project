#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe75  ########


fe75: run
	

build:  $(SRC)/fe75.f
	-$(RM) fe75.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe75.f -o fe75.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe75.$(OBJX) check.$(OBJX) $(LIBS) -o fe75.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe75
	fe75.$(EXESUFFIX)

verify: ;

fe75.run: run

