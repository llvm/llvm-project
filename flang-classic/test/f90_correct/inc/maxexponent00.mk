#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test maxexponent00  ########


maxexponent00: run
	

build:  $(SRC)/maxexponent00.f08
	-$(RM) maxexponent00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/maxexponent00.f08  -o maxexponent00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) maxexponent00.$(OBJX) check.$(OBJX) $(LIBS) -o maxexponent00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test maxexponent00
	maxexponent00.$(EXESUFFIX)

verify: ;

maxexponent00.run: run

