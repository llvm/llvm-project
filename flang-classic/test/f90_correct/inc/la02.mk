#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test la02  ########


la02: la02.$(OBJX)
	

la02.$(OBJX):  $(SRC)/la02.f
	-$(RM) la02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/la02.f -o la02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) la02.$(OBJX) check.$(OBJX) $(LIBS) -o la02.$(EXESUFFIX)


la02.run: la02.$(OBJX)
	@echo ------------------------------------ executing test la02
	la02.$(EXESUFFIX)
run: la02.$(OBJX)
	@echo ------------------------------------ executing test la02
	la02.$(EXESUFFIX)


build:	la02.$(OBJX)


verify:	;
