#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gd00  ########


gd00: run
	

build:  $(SRC)/gd00.f
	-$(RM) gd00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gd00.f -o gd00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gd00.$(OBJX) check.$(OBJX) $(LIBS) -o gd00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gd00
	gd00.$(EXESUFFIX)

verify: ;

gd00.run: run

