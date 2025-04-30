#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test de00  ########


de00: run
	

build:  $(SRC)/de00.f
	-$(RM) de00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/de00.f -o de00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) de00.$(OBJX) check.$(OBJX) $(LIBS) -o de00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test de00
	de00.$(EXESUFFIX)

verify: ;

de00.run: run

