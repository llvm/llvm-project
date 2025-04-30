#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp21  ########


qp21: run


build:  $(SRC)/qp21.f08
	-$(RM) qp21.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp21.f08 -o qp21.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp21.$(OBJX) check.$(OBJX) $(LIBS) -o qp21.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp21
	qp21.$(EXESUFFIX)

verify: ;
