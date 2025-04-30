#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp23  ########


qp23: run


build:  $(SRC)/qp23.f08
	-$(RM) qp23.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp23.f08 -o qp23.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp23.$(OBJX) check.$(OBJX) $(LIBS) -o qp23.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp23
	qp23.$(EXESUFFIX)

verify: ;
