#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp11  ########


qp11: run


build:  $(SRC)/qp11.f08
	-$(RM) qp11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp11.f08 -o qp11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp11.$(OBJX) check.$(OBJX) $(LIBS) -o qp11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp11
	qp11.$(EXESUFFIX)

verify: ;
