#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ee00  ########


ee00: run
	

build:  $(SRC)/ee00.f
	-$(RM) ee00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ee00.f -o ee00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ee00.$(OBJX) check.$(OBJX) $(LIBS) -o ee00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ee00
	ee00.$(EXESUFFIX)

verify: ;

ee00.run: run

