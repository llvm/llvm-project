#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp06  ########


qp06: run


build:  $(SRC)/qp06.f08
	-$(RM) qp06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp06.f08 -o qp06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp06.$(OBJX) check.$(OBJX) $(LIBS) -o qp06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp06
	qp06.$(EXESUFFIX)

verify: ;
