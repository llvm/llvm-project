#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test read  ########


qp41: run


build:  $(SRC)/qp41.f08
	-$(RM) qp41.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.* *.in
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp41.f08 -o qp41.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp41.$(OBJX) check.$(OBJX) $(LIBS) -o qp41.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp41
	qp41.$(EXESUFFIX)

verify: ;


