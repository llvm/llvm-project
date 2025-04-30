#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe110  ########


fe110: run
	

build:  $(SRC)/fe110.f90
	-$(RM) fe110.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe110.f90 -o fe110.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe110.$(OBJX) check.$(OBJX) $(LIBS) -o fe110.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe110
	fe110.$(EXESUFFIX)

verify: ;

fe110.run: run

