#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qtoint  ########


qp63: run


build:  $(SRC)/qp63.f08
	-$(RM) qp63.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp63.f08 -o qp63.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp63.$(OBJX) check.$(OBJX) $(LIBS) -o qp63.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp63
	qp63.$(EXESUFFIX)

verify: ;


