#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test maxexponent01  ########


maxexponent01: run
	

build:  $(SRC)/maxexponent01.f08
	-$(RM) maxexponent01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/maxexponent01.f08 -Mpreprocess -o maxexponent01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) maxexponent01.$(OBJX) check.$(OBJX) $(LIBS) -o maxexponent01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test maxexponent01
	maxexponent01.$(EXESUFFIX)

verify: ;

maxexponent01.run: run

