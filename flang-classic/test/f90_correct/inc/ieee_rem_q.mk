#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test ieee_rem_q  ########


ieee_rem_q: run


build:  $(SRC)/ieee_rem_q.f08
	-$(RM) ieee_rem_q.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee_rem_q.f08 -o ieee_rem_q.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee_rem_q.$(OBJX) check.$(OBJX) $(LIBS) -o ieee_rem_q.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ieee_rem_q
	ieee_rem_q.$(EXESUFFIX)

verify: ;
