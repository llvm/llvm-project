#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ge00  ########


ge00: run
	

build:  $(SRC)/ge00.f
	-$(RM) ge00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ge00.f -o ge00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ge00.$(OBJX) check.$(OBJX) $(LIBS) -o ge00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ge00
	ge00.$(EXESUFFIX)

verify: ;

ge00.run: run

