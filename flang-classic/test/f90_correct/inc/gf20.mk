#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gf20  ########


gf20: run
	

build:  $(SRC)/gf20.f
	-$(RM) gf20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gf20.f -o gf20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gf20.$(OBJX) check.$(OBJX) $(LIBS) -o gf20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gf20
	gf20.$(EXESUFFIX)

verify: ;

gf20.run: run

