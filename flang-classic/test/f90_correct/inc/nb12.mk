#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb12  ########


nb12: run
	

build:  $(SRC)/nb12.f
	-$(RM) nb12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb12.f -o nb12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb12.$(OBJX) check.$(OBJX) $(LIBS) -o nb12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb12
	nb12.$(EXESUFFIX)

verify: ;

nb12.run: run

