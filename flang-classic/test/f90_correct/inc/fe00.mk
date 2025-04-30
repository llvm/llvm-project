#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe00  ########


fe00: run
	

build:  $(SRC)/fe00.f
	-$(RM) fe00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe00.f -o fe00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe00.$(OBJX) check.$(OBJX) $(LIBS) -o fe00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe00
	fe00.$(EXESUFFIX)

verify: ;

fe00.run: run

