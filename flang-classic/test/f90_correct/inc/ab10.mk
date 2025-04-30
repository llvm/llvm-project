#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ab10  ########


ab10: run
	

build:  $(SRC)/ab10.f
	-$(RM) ab10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ab10.f -o ab10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ab10.$(OBJX) check.$(OBJX) $(LIBS) -o ab10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ab10
	ab10.$(EXESUFFIX)

verify: ;

ab10.run: run

