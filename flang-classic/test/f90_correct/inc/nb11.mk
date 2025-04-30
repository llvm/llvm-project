#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb11  ########


nb11: run
	

build:  $(SRC)/nb11.f
	-$(RM) nb11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb11.f -o nb11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb11.$(OBJX) check.$(OBJX) $(LIBS) -o nb11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb11
	nb11.$(EXESUFFIX)

verify: ;

nb11.run: run

