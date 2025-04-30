#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp29  ########


qp29: run


build:  $(SRC)/qp29.f08
	-$(RM) qp29.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp29.f08 -o qp29.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp29.$(OBJX) check.$(OBJX) $(LIBS) -o qp29.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp29
	qp29.$(EXESUFFIX)

verify: ;
