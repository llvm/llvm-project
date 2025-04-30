#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qtoint  ########


qp116: run


build:  $(SRC)/qp116.f08
	-$(RM) qp116.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp116.f08 -o qp116.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp116.$(OBJX) check.$(OBJX) $(LIBS) -o qp116.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp116
	qp116.$(EXESUFFIX)

verify: ;


