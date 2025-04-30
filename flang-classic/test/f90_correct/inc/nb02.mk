#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb02  ########


nb02: run
	

build:  $(SRC)/nb02.f
	-$(RM) nb02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb02.f -o nb02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb02.$(OBJX) check.$(OBJX) $(LIBS) -o nb02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb02
	nb02.$(EXESUFFIX)

verify: ;

nb02.run: run

