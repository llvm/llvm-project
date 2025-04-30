#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee_q3  ########


ieee_q3: run


build:  $(SRC)/ieee_q3.f08
	-$(RM) ieee_q3.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee_q3.f08 -o ieee_q3.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee_q3.$(OBJX) check.$(OBJX) $(LIBS) -o ieee_q3.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ieee_q3
	ieee_q3.$(EXESUFFIX)

verify: ;
