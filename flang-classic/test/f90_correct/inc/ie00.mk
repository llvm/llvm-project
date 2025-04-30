#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ie00  ########


ie00: run
	

build:  $(SRC)/ie00.f
	-$(RM) ie00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ie00.f -o ie00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ie00.$(OBJX) check.$(OBJX) $(LIBS) -o ie00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ie00
	ie00.$(EXESUFFIX)

verify: ;

ie00.run: run

