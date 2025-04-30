#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ge01  ########


ge01: run
	

build:  $(SRC)/ge01.f
	-$(RM) ge01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ge01.f -o ge01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ge01.$(OBJX) check.$(OBJX) $(LIBS) -o ge01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ge01
	ge01.$(EXESUFFIX)

verify: ;

ge01.run: run

