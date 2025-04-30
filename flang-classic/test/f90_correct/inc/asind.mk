#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test asind  ########


asind: run


build:  $(SRC)/asind.f08
	-$(RM) asind.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/asind.f08 -o asind.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) asind.$(OBJX) check.$(OBJX) $(LIBS) -o asind.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test asind
	asind.$(EXESUFFIX)

verify: ;
