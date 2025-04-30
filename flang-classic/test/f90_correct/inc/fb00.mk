#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fb00  ########


fb00: run
	

build:  $(SRC)/fb00.f
	-$(RM) fb00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fb00.f -o fb00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fb00.$(OBJX) check.$(OBJX) $(LIBS) -o fb00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fb00
	fb00.$(EXESUFFIX)

verify: ;

fb00.run: run

