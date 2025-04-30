#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qtoint  ########


qp85: run


build:  $(SRC)/qp85.f08
	-$(RM) qp85.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp85.f08 -o qp85.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp85.$(OBJX) check.$(OBJX) $(LIBS) -o qp85.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp85
	qp85.$(EXESUFFIX)

verify: ;


