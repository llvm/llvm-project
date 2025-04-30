#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gf00  ########


gf00: run
	

build:  $(SRC)/gf00.f
	-$(RM) gf00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gf00.f -o gf00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gf00.$(OBJX) check.$(OBJX) $(LIBS) -o gf00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gf00
	gf00.$(EXESUFFIX)

verify: ;

gf00.run: run

