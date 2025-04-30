#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp22  ########


qp22: run


build:  $(SRC)/qp22.f08
	-$(RM) qp22.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp22.f08 -o qp22.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp22.$(OBJX) check.$(OBJX) $(LIBS) -o qp22.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp22
	qp22.$(EXESUFFIX)

verify: ;
