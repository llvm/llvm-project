#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bc00  ########


bc00: run
	

build:  $(SRC)/bc00.f
	-$(RM) bc00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bc00.f -o bc00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bc00.$(OBJX) check.$(OBJX) $(LIBS) -o bc00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bc00
	bc00.$(EXESUFFIX)

verify: ;

bc00.run: run

