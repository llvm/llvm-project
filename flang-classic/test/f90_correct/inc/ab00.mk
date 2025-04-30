#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ab00  ########


ab00: run
	

build:  $(SRC)/ab00.f
	-$(RM) ab00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ab00.f -o ab00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ab00.$(OBJX) check.$(OBJX) $(LIBS) -o ab00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ab00
	ab00.$(EXESUFFIX)

verify: ;

ab00.run: run

