#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe20  ########


fe20: run
	

build:  $(SRC)/fe20.f90
	-$(RM) fe20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe20.f90 -o fe20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe20.$(OBJX) check.$(OBJX) $(LIBS) -o fe20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe20
	fe20.$(EXESUFFIX)

verify: ;

fe20.run: run

