#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp14  ########


qp14: run


build:  $(SRC)/qp14.f08
	-$(RM) qp14.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp14.f08 -o qp14.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp14.$(OBJX) check.$(OBJX) $(LIBS) -o qp14.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp14
	qp14.$(EXESUFFIX)

verify: ;
