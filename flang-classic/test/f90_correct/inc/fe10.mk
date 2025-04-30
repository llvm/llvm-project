#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe10  ########


fe10: run
	

build:  $(SRC)/fe10.f
	-$(RM) fe10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe10.f -o fe10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe10.$(OBJX) check.$(OBJX) $(LIBS) -o fe10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe10
	fe10.$(EXESUFFIX)

verify: ;

fe10.run: run

