#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp08  ########


qp08: run


build:  $(SRC)/qp08.f08
	-$(RM) qp08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp08.f08 -o qp08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp08.$(OBJX) check.$(OBJX) $(LIBS) -o qp08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp08
	qp08.$(EXESUFFIX)

verify: ;
