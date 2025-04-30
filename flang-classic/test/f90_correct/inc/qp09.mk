#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp09  ########


qp09: run


build:  $(SRC)/qp09.f08
	-$(RM) qp09.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp09.f08 -o qp09.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp09.$(OBJX) check.$(OBJX) $(LIBS) -o qp09.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp09
	qp09.$(EXESUFFIX)

verify: ;
