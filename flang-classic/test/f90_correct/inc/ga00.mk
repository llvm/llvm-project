#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ga00  ########


ga00: run
	

build:  $(SRC)/ga00.f
	-$(RM) ga00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ga00.f -o ga00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ga00.$(OBJX) check.$(OBJX) $(LIBS) -o ga00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ga00
	ga00.$(EXESUFFIX)

verify: ;

ga00.run: run

