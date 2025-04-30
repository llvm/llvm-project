#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test acosd  ########


acosd: run


build:  $(SRC)/acosd.f08
	-$(RM) acosd.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/acosd.f08 -o acosd.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) acosd.$(OBJX) check.$(OBJX) $(LIBS) -o acosd.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test acosd
	acosd.$(EXESUFFIX)

verify: ;
