#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp28  ########


qp28: run


build:  $(SRC)/qp28.f08
	-$(RM) qp28.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp28.f08 -o qp28.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp28.$(OBJX) check.$(OBJX) $(LIBS) -o qp28.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp28
	qp28.$(EXESUFFIX)

verify: ;
