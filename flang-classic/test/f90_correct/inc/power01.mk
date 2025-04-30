#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test power01  ########


power01: run


build:  $(SRC)/power01.f08
	-$(RM) power01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/power01.f08 -o power01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) power01.$(OBJX) check.$(OBJX) $(LIBS) -o power01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test power01
	power01.$(EXESUFFIX)

verify: ;
