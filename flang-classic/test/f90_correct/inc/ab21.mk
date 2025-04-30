#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ab21  ########


ab21: run
	

build:  $(SRC)/ab21.f
	-$(RM) ab21.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ab21.f -o ab21.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ab21.$(OBJX) check.$(OBJX) $(LIBS) -o ab21.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ab21
	ab21.$(EXESUFFIX)

verify: ;

ab21.run: run

