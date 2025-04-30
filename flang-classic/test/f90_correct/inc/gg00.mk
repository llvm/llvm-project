#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gg00  ########


gg00: run
	

build:  $(SRC)/gg00.f
	-$(RM) gg00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gg00.f -o gg00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gg00.$(OBJX) check.$(OBJX) $(LIBS) -o gg00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gg00
	gg00.$(EXESUFFIX)

verify: ;

gg00.run: run

