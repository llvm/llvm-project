#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gf10  ########


gf10: run
	

build:  $(SRC)/gf10.f
	-$(RM) gf10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gf10.f -o gf10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gf10.$(OBJX) check.$(OBJX) $(LIBS) -o gf10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gf10
	gf10.$(EXESUFFIX)

verify: ;

gf10.run: run

