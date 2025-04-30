#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp13  ########


qp13: run


build:  $(SRC)/qp13.f08
	-$(RM) qp13.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp13.f08 -o qp13.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp13.$(OBJX) check.$(OBJX) $(LIBS) -o qp13.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp13
	qp13.$(EXESUFFIX)

verify: ;
