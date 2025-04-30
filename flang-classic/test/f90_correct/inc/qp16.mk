#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp16  ########


qp16: run


build:  $(SRC)/qp16.f08
	-$(RM) qp16.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp16.f08 -o qp16.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp16.$(OBJX) check.$(OBJX) $(LIBS) -o qp16.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp16
	qp16.$(EXESUFFIX)

verify: ;
