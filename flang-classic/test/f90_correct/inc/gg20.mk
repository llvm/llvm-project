#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gg20  ########


gg20: run
	

build:  $(SRC)/gg20.f
	-$(RM) gg20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gg20.f -o gg20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gg20.$(OBJX) check.$(OBJX) $(LIBS) -o gg20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gg20
	gg20.$(EXESUFFIX)

verify: ;

gg20.run: run

