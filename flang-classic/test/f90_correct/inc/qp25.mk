#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp25  ########


qp25: run


build:  $(SRC)/qp25.f08
	-$(RM) qp25.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp25.f08 -o qp25.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp25.$(OBJX) check.$(OBJX) $(LIBS) -o qp25.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp25
	qp25.$(EXESUFFIX)

verify: ;
