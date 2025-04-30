#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ab20  ########


ab20: run
	

build:  $(SRC)/ab20.f
	-$(RM) ab20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ab20.f -o ab20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ab20.$(OBJX) check.$(OBJX) $(LIBS) -o ab20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ab20
	ab20.$(EXESUFFIX)

verify: ;

ab20.run: run

